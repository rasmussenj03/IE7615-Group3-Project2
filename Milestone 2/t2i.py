"""Text-to-image pipeline for Milestone 2.

Stable Diffusion 1.5 is loaded as separate components (CLIP tokenizer, CLIP text
encoder, UNet, VAE, noise scheduler) and wired together by hand, so every stage
of the text -> embedding -> conditioning -> image flow is visible:

    prompt --tokenizer--> token ids [B, 77]
           --CLIP text encoder--> per-token embeddings c [B, 77, 768]
    z_T ~ N(0, I) [B, 4, 64, 64]  (one seeded CPU generator per image)
    for t in scheduler.timesteps:
        eps_u, eps_c = UNet(z_t, t, context=[c_empty, c])   # cross-attention on c
        eps = eps_u + w * (eps_c - eps_u)                     # classifier-free guidance
        z_{t-1} = scheduler.step(eps, t, z_t)                 # DDPM / DDIM / DPM-Solver ...
    image = VAE.decode(z_0 / 0.18215)                         # [B, 3, 512, 512]

The same file holds the sampler registry used by the sweeps, a CLIP ViT-B/32
scorer (the Milestone 1 model) for prompt-alignment and diversity checks, and
helpers that save every image with its prompt, seed and settings.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_EVAL_ID = "openai/clip-vit-base-patch32"  # Milestone 1 encoder, used only for scoring


# ---------------------------------------------------------------------------
# Device and environment
# ---------------------------------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    # fp16 halves memory and roughly doubles speed on CUDA; CPU and MPS stay in fp32
    return torch.float16 if device == "cuda" else torch.float32


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: str) -> float | None:
    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return None


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["torch", "diffusers", "transformers", "peft", "accelerate", "safetensors"]:
        try:
            info[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info[name] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 2**30, 1)
        info["cuda"] = torch.version.cuda
    return info


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

@dataclass
class SDComponents:
    tokenizer: Any
    text_encoder: Any
    unet: Any
    vae: Any
    scheduler_config: dict
    device: str
    dtype: torch.dtype
    model_id: str = MODEL_ID

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    @property
    def latent_scaling(self) -> float:
        return float(self.vae.config.scaling_factor)


def load_components(model_id: str = MODEL_ID, device: str | None = None,
                    dtype: torch.dtype | None = None) -> SDComponents:
    """Load the four Stable Diffusion parts separately, frozen and in eval mode."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    device = device or pick_device()
    dtype = dtype or default_dtype(device)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=True)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
    # The scheduler config holds the training noise schedule (scaled_linear, 1000 steps)
    scheduler_config = dict(DDIMScheduler.load_config(model_id, subfolder="scheduler"))

    for module in (text_encoder, unet, vae):
        module.to(device, dtype).eval().requires_grad_(False)

    return SDComponents(tokenizer, text_encoder, unet, vae, scheduler_config, device, dtype, model_id)


def cross_attention_layers(unet) -> list[dict]:
    """Every cross-attention block (named attn2) in the UNet, with its dimensions."""
    layers = []
    for name, module in unet.named_modules():
        if name.endswith("attn2") and hasattr(module, "to_k"):
            layers.append({
                "name": name,
                "stage": name.split(".")[0],
                "query_dim": module.to_q.in_features,
                "context_dim": module.to_k.in_features,
                "heads": module.heads,
            })
    return layers


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe_components(c: SDComponents) -> dict:
    """Sizes and shapes of each stage, for the architecture table in the write-up."""
    xattn = cross_attention_layers(c.unet)
    h = w = c.unet.config.sample_size * c.vae_scale_factor
    return {
        "model_id": c.model_id,
        "device": c.device,
        "dtype": str(c.dtype),
        "tokenizer_max_length": c.tokenizer.model_max_length,
        "text_encoder": type(c.text_encoder).__name__,
        "text_encoder_params_M": round(count_params(c.text_encoder) / 1e6, 1),
        "text_hidden_size": c.text_encoder.config.hidden_size,
        "unet_params_M": round(count_params(c.unet) / 1e6, 1),
        "unet_cross_attention_dim": c.unet.config.cross_attention_dim,
        "unet_cross_attention_layers": len(xattn),
        "vae_params_M": round(count_params(c.vae) / 1e6, 1),
        "vae_scaling_factor": c.latent_scaling,
        "latent_shape": [c.unet.config.in_channels, h // c.vae_scale_factor, w // c.vae_scale_factor],
        "image_size": [h, w],
        "train_noise_schedule": {k: c.scheduler_config.get(k) for k in
                                 ["num_train_timesteps", "beta_schedule", "beta_start",
                                  "beta_end", "prediction_type"]},
    }


# ---------------------------------------------------------------------------
# Text -> embedding
# ---------------------------------------------------------------------------

def tokenize(c: SDComponents, prompts: Sequence[str]):
    return c.tokenizer(list(prompts), padding="max_length", max_length=c.tokenizer.model_max_length,
                       truncation=True, return_tensors="pt")


def token_report(c: SDComponents, prompts: Sequence[str]) -> list[dict]:
    """Token counts per prompt, flagging prompts CLIP truncates at 77 tokens."""
    limit = c.tokenizer.model_max_length
    out = []
    for p in prompts:
        n = len(c.tokenizer(p).input_ids)
        out.append({"prompt": p, "tokens": n, "truncated": n > limit})
    return out


@torch.no_grad()
def encode_prompt(c: SDComponents, prompts: str | Sequence[str]) -> torch.Tensor:
    """Per-token CLIP hidden states [B, 77, 768], the UNet's cross-attention context.

    Stable Diffusion uses the last hidden state of the text transformer, not the
    pooled and projected sentence embedding that Milestone 1 used for retrieval.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    ids = tokenize(c, prompts).input_ids.to(c.device)
    return c.text_encoder(ids)[0].to(c.dtype)


# ---------------------------------------------------------------------------
# Samplers and noise schedules
# ---------------------------------------------------------------------------

SAMPLERS: dict[str, tuple[str, dict]] = {
    "ddpm": ("DDPMScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "pndm": ("PNDMScheduler", {"skip_prk_steps": True}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}),
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler",
                        {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}),
}

STOCHASTIC_SAMPLERS = {"ddpm", "euler_a"}  # inject fresh noise at every step


def make_scheduler(name: str, base_config: dict, **overrides):
    """Build a sampler from the model's training schedule, with optional overrides
    (for example beta_schedule or timestep_spacing for the noise-schedule tests)."""
    import diffusers

    cls_name, defaults = SAMPLERS[name]
    cls = getattr(diffusers, cls_name)
    return cls.from_config(base_config, **{**defaults, **overrides})


def schedule_curves(base_config: dict, schedules: dict[str, dict] | None = None) -> dict[str, dict]:
    """alpha_bar_t and log-SNR for several beta schedules, for the theory figure."""
    from diffusers import DDPMScheduler

    schedules = schedules or {
        "scaled_linear (SD 1.5 training)": {"beta_schedule": "scaled_linear"},
        "linear, same endpoints": {"beta_schedule": "linear"},
        "linear, DDPM paper (1e-4 to 0.02)": {"beta_schedule": "linear", "beta_start": 1e-4, "beta_end": 0.02},
        "cosine (squaredcos_cap_v2)": {"beta_schedule": "squaredcos_cap_v2"},
    }
    out = {}
    for label, kw in schedules.items():
        s = DDPMScheduler.from_config(base_config, **kw)
        ab = s.alphas_cumprod.double()
        out[label] = {"alpha_bar": ab.numpy(), "log_snr": torch.log(ab / (1 - ab)).numpy(), "overrides": kw}
    return out


# ---------------------------------------------------------------------------
# Image <-> latent
# ---------------------------------------------------------------------------

def initial_latents(c: SDComponents, seeds: Sequence[int], height: int, width: int):
    """One CPU generator per image, so a seed gives the same z_T on any GPU and in any
    batch. The draw order matches diffusers' randn_tensor for a list of generators."""
    shape = (1, c.unet.config.in_channels, height // c.vae_scale_factor, width // c.vae_scale_factor)
    generators = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
    latents = torch.cat([torch.randn(shape, generator=g, dtype=c.dtype) for g in generators])
    return latents.to(c.device), generators


@torch.no_grad()
def decode_latents(c: SDComponents, latents: torch.Tensor) -> list[Image.Image]:
    images = c.vae.decode(latents.to(c.dtype) / c.latent_scaling).sample
    images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return [Image.fromarray((im * 255).round().astype("uint8")) for im in images]


@torch.no_grad()
def encode_images(c: SDComponents, pixels: torch.Tensor, sample: bool = True) -> torch.Tensor:
    """pixels in [-1, 1], [B, 3, H, W] -> scaled latents [B, 4, H/8, W/8]."""
    dist = c.vae.encode(pixels.to(c.device, c.dtype)).latent_dist
    z = dist.sample() if sample else dist.mean
    return z * c.latent_scaling


# ---------------------------------------------------------------------------
# Conditional generation
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    images: list[Image.Image]
    settings: dict
    timings: dict
    peak_memory_mb: float | None
    diagnostics: list[dict] = field(default_factory=list)
    x0_previews: dict[int, list[Image.Image]] = field(default_factory=dict)
    timesteps: list[int] = field(default_factory=list)


@torch.no_grad()
def generate(
    c: SDComponents,
    prompts: str | Sequence[str],
    seeds: int | Sequence[int],
    sampler: str = "ddim",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    negative_prompt: str = "",
    eta: float = 0.0,
    scheduler_overrides: dict | None = None,
    prompt_embeds: torch.Tensor | None = None,
    track: bool = False,
    x0_preview_steps: Iterable[int] = (),
) -> GenerationResult:
    """Denoise from seeded noise to images, conditioned on the prompts.

    guidance_scale == 1 runs the conditional branch only (one UNet pass per step);
    any other value runs both branches and applies classifier-free guidance,
    eps = eps_uncond + w * (eps_cond - eps_uncond). w = 0 is unconditional sampling.
    prompt_embeds replaces the text encoder output (used for the ablations).
    track=True records per-step sampler diagnostics.
    """
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    seeds = [seeds] * len(prompts) if isinstance(seeds, int) else list(seeds)
    assert len(prompts) == len(seeds), "one seed per prompt"
    batch = len(prompts)
    scheduler = make_scheduler(sampler, c.scheduler_config, **(scheduler_overrides or {}))
    x0_preview_steps = set(x0_preview_steps)

    sync(c.device)
    reset_peak_memory(c.device)
    t_start = time.perf_counter()

    # 1. Text -> conditioning context
    cond = encode_prompt(c, prompts) if prompt_embeds is None else prompt_embeds.to(c.device, c.dtype)
    do_cfg = guidance_scale != 1.0
    if do_cfg:
        uncond = encode_prompt(c, [negative_prompt] * batch)
        context = torch.cat([uncond, cond])  # same [uncond, cond] order as diffusers
    else:
        context = cond
    sync(c.device)
    t_encoded = time.perf_counter()

    # 2. Seeded Gaussian noise z_T
    latents, generators = initial_latents(c, seeds, height, width)
    scheduler.set_timesteps(num_inference_steps, device=c.device)
    latents = latents * scheduler.init_noise_sigma

    step_params = set(inspect.signature(scheduler.step).parameters)
    extra = {}
    if "eta" in step_params:
        extra["eta"] = eta
    if "generator" in step_params:
        extra["generator"] = generators  # stochastic samplers draw step noise from the same seeds

    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    diagnostics, previews = [], {}

    # 3. Reverse diffusion with classifier-free guidance
    for i, t in enumerate(scheduler.timesteps):
        t_step = time.perf_counter()
        model_in = torch.cat([latents] * 2) if do_cfg else latents
        model_in = scheduler.scale_model_input(model_in, t)
        eps = c.unet(model_in, t, encoder_hidden_states=context).sample
        if do_cfg:
            eps_u, eps_c = eps.chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        out = scheduler.step(eps, t, latents, **extra)

        if track or i in x0_preview_steps:
            x0 = getattr(out, "pred_original_sample", None)
            if x0 is None and alphas_cumprod is not None:
                # VP form: z_t = sqrt(ab) x0 + sqrt(1 - ab) eps
                ab = alphas_cumprod[int(t)].to(latents.device, torch.float32)
                x0 = (latents.float() - (1 - ab).sqrt() * eps.float()) / ab.sqrt()
            if track:
                sync(c.device)
                row = {
                    "step": i,
                    "timestep": int(t),
                    "latent_std": latents.float().std().item(),
                    "eps_rms": eps.float().pow(2).mean().sqrt().item(),
                    "step_seconds": time.perf_counter() - t_step,
                }
                if do_cfg:
                    # How much the text changes the noise prediction at this step
                    row["guidance_delta_rms"] = (eps_c - eps_u).float().pow(2).mean().sqrt().item()
                diagnostics.append(row)
            if i in x0_preview_steps and x0 is not None:
                previews[i] = decode_latents(c, x0)

        latents = out.prev_sample

    sync(c.device)
    t_denoised = time.perf_counter()

    # 4. Latent -> image
    images = decode_latents(c, latents)
    sync(c.device)
    t_end = time.perf_counter()

    settings = {
        "sampler": sampler,
        "scheduler_class": type(scheduler).__name__,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "negative_prompt": negative_prompt,
        "eta": eta if "eta" in extra else None,
        "scheduler_overrides": scheduler_overrides or {},
        "unet_passes_per_step": 2 if do_cfg else 1,
    }
    timings = {
        "encode_s": round(t_encoded - t_start, 3),
        "denoise_s": round(t_denoised - t_encoded, 3),
        "decode_s": round(t_end - t_denoised, 3),
        "total_s": round(t_end - t_start, 3),
        "per_image_s": round((t_end - t_start) / batch, 3),
        "batch_size": batch,
    }
    return GenerationResult(images, settings, timings, peak_memory_mb(c.device), diagnostics,
                            previews, [int(t) for t in scheduler.timesteps])


# ---------------------------------------------------------------------------
# Cross-attention maps: where in the image each prompt token is read
# ---------------------------------------------------------------------------

class _StoreCrossAttention:
    """Plain (non-fused) attention that also accumulates the conditional branch's
    cross-attention probabilities at one spatial resolution."""

    def __init__(self, store: dict, res: int, do_cfg: bool):
        self.store, self.res, self.do_cfg = store, res, do_cfg

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        b, seq, _ = hidden_states.shape
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(context))
        v = attn.head_to_batch_dim(attn.to_v(context))
        probs = attn.get_attention_scores(q, k, attention_mask)  # [b * heads, seq, 77]
        if seq == self.res * self.res:
            p = probs.view(b, attn.heads, seq, -1)
            p = p[b // 2:] if self.do_cfg else p  # keep the text-conditioned half
            self.store["sum"] = self.store.get("sum", 0) + p.float().mean(1).sum(0).cpu()
            self.store["count"] = self.store.get("count", 0) + p.shape[0]
        out = attn.batch_to_head_dim(torch.bmm(probs, v))
        return attn.to_out[1](attn.to_out[0](out))


@torch.no_grad()
def cross_attention_maps(c: SDComponents, prompt: str, seed: int, sampler: str = "ddim",
                         num_inference_steps: int = 30, guidance_scale: float = 7.5, res: int = 16):
    """Generate one image and return (image, tokens, maps [n_tokens, res, res]),
    averaging cross-attention over heads, layers at that resolution, and steps."""
    store: dict = {}
    original = c.unet.attn_processors
    procs = {name: (_StoreCrossAttention(store, res, guidance_scale != 1.0) if "attn2" in name else proc)
             for name, proc in original.items()}
    c.unet.set_attn_processor(procs)
    try:
        result = generate(c, [prompt], [seed], sampler, num_inference_steps, guidance_scale)
    finally:
        c.unet.set_attn_processor(original)
    if "sum" not in store:
        raise ValueError(f"no cross-attention layer runs at {res}x{res}; pick another res")
    ids =c.tokenizer(prompt, truncation=True, max_length=c.tokenizer.model_max_length).input_ids
    tokens = c.tokenizer.convert_ids_to_tokens(ids)[1:-1]  # drop start and end tokens
    maps = (store["sum"] / store["count"]).T[1:1 + len(tokens)].reshape(len(tokens), res, res)
    return result.images[0], [t.replace("</w>", "") for t in tokens], maps.numpy()


# ---------------------------------------------------------------------------
# Batched, resumable runs that save every image with its metadata
# ---------------------------------------------------------------------------

def job_name(job: dict) -> str:
    g = f"{float(job['guidance_scale']):g}".replace(".", "p")
    name = f"{job['prompt_id']}_seed{job['seed']}_{job['sampler']}_n{job['steps']}_g{g}"
    return name + (f"_{job['variant']}" if job.get("variant") else "")


def run_jobs(c: SDComponents, jobs: list[dict], out_dir: Path, batch_size: int = 4,
             image_format: str = "jpg", extra_meta: dict | None = None, verbose: bool = True) -> list[dict]:
    """Generate and save each job (prompt_id, prompt, seed, sampler, steps, guidance_scale,
    optional variant and scheduler_overrides). Jobs that already have an image and a JSON
    file are skipped, so a run can resume after a Colab disconnect. Returns every job's
    metadata record and writes them all to out_dir / manifest.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_info()

    def settings_key(j):
        return (j["sampler"], j["steps"], float(j["guidance_scale"]),
                json.dumps(j.get("scheduler_overrides", {}), sort_keys=True), j.get("negative_prompt", ""))

    todo = [j for j in jobs if not (out_dir / f"{job_name(j)}.json").exists()]
    groups: dict[tuple, list[dict]] = {}
    for j in todo:
        groups.setdefault(settings_key(j), []).append(j)

    done = 0
    for key, group in groups.items():
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            first = chunk[0]
            res = generate(c, [j["prompt"] for j in chunk], [j["seed"] for j in chunk],
                           sampler=first["sampler"], num_inference_steps=first["steps"],
                           guidance_scale=first["guidance_scale"],
                           negative_prompt=first.get("negative_prompt", ""),
                           scheduler_overrides=first.get("scheduler_overrides"))
            for j, img in zip(chunk, res.images):
                name = job_name(j)
                img_path = out_dir / f"{name}.{image_format}"
                if image_format == "jpg":
                    img.save(img_path, quality=92)
                else:
                    img.save(img_path)
                meta = {**{k: v for k, v in j.items()}, "image": img_path.name, "model_id": c.model_id,
                        "dtype": str(c.dtype), "settings": res.settings, "timings": res.timings,
                        "peak_memory_mb": res.peak_memory_mb, "environment": env, **(extra_meta or {})}
                (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            done += len(chunk)
            if verbose:
                print(f"  {done}/{len(todo)} new | {first['sampler']} n={first['steps']} "
                      f"w={first['guidance_scale']} | {res.timings['per_image_s']:.2f} s/image")

    records = [json.loads((out_dir / f"{job_name(j)}.json").read_text(encoding="utf-8")) for j in jobs]
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    if verbose:
        print(f"{len(records)} images in {out_dir} ({len(records) - len(todo)} reused)")
    return records


def load_image(out_dir: Path, record: dict) -> Image.Image:
    return Image.open(Path(out_dir) / record["image"]).convert("RGB")


# ---------------------------------------------------------------------------
# CLIP ViT-B/32 scoring (prompt alignment, retrieval, diversity)
# ---------------------------------------------------------------------------

class ClipScorer:
    """Scores generated images with the Milestone 1 CLIP model. It is a different
    CLIP than the one conditioning Stable Diffusion (ViT-L/14 text tower), so the
    score is not just the generator grading its own text encoder."""

    def __init__(self, model_id: str = CLIP_EVAL_ID, device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or pick_device()
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def image_embeds(self, images: Sequence[Image.Image], batch_size: int = 32) -> torch.Tensor:
        chunks = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=list(images[i:i + batch_size]), return_tensors="pt").to(self.device)
            chunks.append(torch.nn.functional.normalize(self.model.get_image_features(**inputs), dim=-1).cpu())
        return torch.cat(chunks)

    @torch.no_grad()
    def text_embeds(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        chunks = []
        for i in range(0, len(texts), batch_size):
            inputs = self.processor.tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                                              return_tensors="pt").to(self.device)
            chunks.append(torch.nn.functional.normalize(self.model.get_text_features(**inputs), dim=-1).cpu())
        return torch.cat(chunks)

    def clip_score(self, images: Sequence[Image.Image], texts: Sequence[str]) -> np.ndarray:
        """100 * cosine(image, its own text), one value per pair."""
        return (100 * (self.image_embeds(images) * self.text_embeds(texts)).sum(-1)).numpy()


def retrieval_top1(image_emb: torch.Tensor, text_emb: torch.Tensor, targets: Sequence[int]) -> float:
    """Share of images whose own prompt is the most similar among all candidate prompts."""
    pred = (image_emb @ text_emb.T).argmax(dim=1)
    return (pred == torch.as_tensor(targets)).float().mean().item()


def pairwise_diversity(image_emb: torch.Tensor) -> float:
    """Mean (1 - cosine) over all pairs: 0 means identical images."""
    n = image_emb.shape[0]
    if n < 2:
        return float("nan")
    sim = image_emb @ image_emb.T
    off = sim[~torch.eye(n, dtype=torch.bool)]
    return (1 - off).mean().item()


def mean_saturation(img: Image.Image) -> float:
    """Mean HSV saturation in [0, 1]; rises when high guidance over-saturates colors."""
    return float(np.asarray(img.convert("HSV"))[..., 1].mean() / 255)


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def labeled_grid(images: list[list[Image.Image | None]], row_labels: Sequence[str] = (),
                 col_labels: Sequence[str] = (), title: str | None = None, cell: float = 2.4,
                 row_label_width: int = 28):
    """Matplotlib grid of images with row and column labels; returns the figure."""
    import textwrap

    import matplotlib.pyplot as plt

    rows, cols = len(images), max(len(r) for r in images)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols + (1.6 if row_labels else 0), cell * rows + 0.6),
                             squeeze=False)
    for r in range(rows):
        for k in range(cols):
            ax = axes[r][k]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            img = images[r][k] if k < len(images[r]) else None
            if img is not None:
                ax.imshow(img)
            if r == 0 and k < len(col_labels):
                ax.set_title(col_labels[k], fontsize=9)
            if k == 0 and r < len(row_labels):
                ax.set_ylabel(textwrap.fill(row_labels[r], row_label_width), fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=8)
    if title:
        fig.suptitle(title, fontsize=11)
    height = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / height) if title else None)  # keep the title clear of column labels
    return fig


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
