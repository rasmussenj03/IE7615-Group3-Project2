"""LoRA fine-tuning of Stable Diffusion's cross-attention on the curated COCO subset.

Only the cross-attention projections (the attn2 blocks, where the UNet reads the
CLIP text embeddings) receive trainable low-rank adapters. The text encoder, VAE
and every other UNet weight stay frozen, so the fine-tune changes how the text
conditions the image and nothing else.

Training objective (epsilon prediction, DDPM forward process of the base model):
    z0 = VAE.encode(image) * 0.18215
    t ~ U{0, ..., 999},  eps ~ N(0, I)
    z_t = sqrt(alpha_bar_t) z0 + sqrt(1 - alpha_bar_t) eps
    loss = || eps - UNet(z_t, t, CLIP(caption)) ||^2
Captions are dropped to "" with probability caption_dropout, so the
unconditional branch that classifier-free guidance relies on keeps training too.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from t2i import (SDComponents, encode_images, encode_prompt, environment_info,
                 generate, peak_memory_mb, reset_peak_memory, sync)

CROSS_ATTENTION_TARGETS = ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"]
COCO_URL = "http://images.cocodataset.org/train2017/{}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def preprocess(img: Image.Image, size: int = 512) -> Image.Image:
    """Shortest side to `size`, then center crop: the Milestone 1 export transform."""
    from torchvision import transforms

    return transforms.CenterCrop(size)(transforms.Resize(size)(img.convert("RGB")))


def ensure_images(records: list[dict], image_root: Path, size: int = 512, workers: int = 16) -> list[dict]:
    """Return the records whose image exists under image_root, downloading and
    preprocessing any missing ones from cocodataset.org. This avoids re-running the
    FiftyOne download on a fresh Colab runtime."""
    image_root = Path(image_root)

    def fetch(r):
        path = image_root / r["image"]
        if path.exists():
            return r
        try:
            req = urllib.request.Request(COCO_URL.format(r["file_name"]), headers={"User-Agent": "IE7615-project"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                from io import BytesIO
                img = Image.open(BytesIO(resp.read()))
            path.parent.mkdir(parents=True, exist_ok=True)
            preprocess(img, size).save(path)
            return r
        except Exception as err:  # keep going; report the count at the end
            print(f"  skipped {r['file_name']}: {err}")
            return None

    with ThreadPoolExecutor(workers) as pool:
        ready = [r for r in pool.map(fetch, records) if r is not None]
    print(f"Images ready: {len(ready)}/{len(records)} under {image_root}")
    return ready


class CocoCaptionDataset(Dataset):
    """512x512 COCO images in [-1, 1] with one random caption per draw."""

    def __init__(self, records: list[dict], image_root: Path, size: int = 512, caption_dropout: float = 0.1):
        self.records = records
        self.image_root = Path(image_root)
        self.size = size
        self.caption_dropout = caption_dropout

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.image_root / r["image"]).convert("RGB")
        if img.size != (self.size, self.size):
            img = preprocess(img, self.size)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1
        caption = "" if random.random() < self.caption_dropout else random.choice(r["captions"])
        return x, caption


def collate(batch):
    xs, caps = zip(*batch)
    return torch.stack(xs), list(caps)


def split_records(records: list[dict], held_out_image_ids: set[int], n_train: int | None, seed: int = 42):
    """Hold out the baseline-prompt images so the fine-tune never sees them."""
    pool = [r for r in records if r.get("image_id") not in held_out_image_ids and r.get("captions")]
    held = [r for r in records if r.get("image_id") in held_out_image_ids]
    random.Random(seed).shuffle(pool)
    return (pool[:n_train] if n_train else pool), held


# ---------------------------------------------------------------------------
# LoRA adapters
# ---------------------------------------------------------------------------

def add_lora(unet, rank: int = 8, alpha: float | None = None, target_modules=CROSS_ATTENTION_TARGETS):
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    config = LoraConfig(r=rank, lora_alpha=alpha or rank, init_lora_weights="gaussian",
                        target_modules=list(target_modules))
    unet.requires_grad_(False)
    unet.add_adapter(config)
    cast_training_params(unet, dtype=torch.float32)  # adapters train in fp32, base stays fp16
    return config


def lora_parameters(unet):
    return [p for p in unet.parameters() if p.requires_grad]


def save_lora(unet, out_dir: Path, meta: dict) -> Path:
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in get_peft_model_state_dict(unet).items()}
    save_file(state, out_dir / "unet_cross_attention_lora.safetensors")
    (out_dir / "lora_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out_dir


def load_lora(unet, lora_dir: Path, dtype: torch.dtype | None = None):
    """Attach saved adapters to a frozen UNet. Toggle them with unet.disable_adapters()
    and unet.enable_adapters() to compare against the base model in one session."""
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    lora_dir = Path(lora_dir)
    meta = json.loads((lora_dir / "lora_config.json").read_text(encoding="utf-8"))
    add_lora(unet, rank=meta["rank"], alpha=meta["alpha"], target_modules=meta["target_modules"])
    state = load_file(lora_dir / "unet_cross_attention_lora.safetensors")
    result = set_peft_model_state_dict(unet, state)
    unexpected = getattr(result, "unexpected_keys", None)
    assert not unexpected, f"unexpected LoRA keys: {unexpected[:5]}"
    unet.requires_grad_(False)
    if dtype is not None:
        unet.to(dtype)
    return meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    max_steps: int = 1500
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 1e-4
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # cosine decay floor
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    rank: int = 8
    alpha: float = 8
    target_modules: list = field(default_factory=lambda: list(CROSS_ATTENTION_TARGETS))
    caption_dropout: float = 0.1
    resolution: int = 512
    seed: int = 42
    gradient_checkpointing: bool = True
    num_workers: int = 2
    log_every: int = 10
    val_every: int = 100
    sample_every: int = 500
    sample_steps: int = 30
    val_timesteps: tuple = (100, 300, 500, 700, 900)


def lr_lambda(cfg: TrainConfig):
    def f(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return f


class FixedValidation:
    """Validation loss on held-out images with fixed captions, timesteps and noise, so
    the curve is not dominated by the random t and eps draws of single batches."""

    def __init__(self, c: SDComponents, records: list[dict], image_root: Path, timesteps, noise_scheduler,
                 size: int = 512, seed: int = 0):
        g = torch.Generator("cpu").manual_seed(seed)
        ds = CocoCaptionDataset(records, image_root, size, caption_dropout=0.0)
        pixels = torch.stack([ds[i][0] for i in range(len(ds))])
        captions = [r["captions"][0] for r in records]
        self.latents = torch.cat([encode_images(c, pixels[i:i + 4], sample=False) for i in range(0, len(pixels), 4)])
        self.context = torch.cat([encode_prompt(c, captions[i:i + 8]) for i in range(0, len(captions), 8)])
        n, T = len(records), len(timesteps)
        self.t = torch.tensor(list(timesteps) * n, device=c.device)
        self.idx = torch.arange(n).repeat_interleave(T)
        self.noise = torch.randn((n * T, *self.latents.shape[1:]), generator=g).to(c.device, c.dtype)
        self.timesteps = list(timesteps)
        self.noise_scheduler = noise_scheduler

    @torch.no_grad()
    def __call__(self, c: SDComponents, batch_size: int = 8) -> dict:
        losses = []
        for i in range(0, len(self.t), batch_size):
            idx = self.idx[i:i + batch_size]
            z0 = self.latents[idx]
            eps = self.noise[i:i + batch_size]
            t = self.t[i:i + batch_size]
            zt = self.noise_scheduler.add_noise(z0, eps, t)
            with torch.autocast("cuda", dtype=torch.float16, enabled=c.device == "cuda"):
                pred = c.unet(zt, t, encoder_hidden_states=self.context[idx]).sample
            losses.append(F.mse_loss(pred.float(), eps.float(), reduction="none").mean(dim=(1, 2, 3)).cpu())
        per = torch.cat(losses)
        out = {"val_loss": per.mean().item()}
        T = len(self.timesteps)
        for k, t in enumerate(self.timesteps):
            out[f"val_loss_t{t}"] = per[k::T].mean().item()
        return out


def train_lora(c: SDComponents, train_records: list[dict], val_records: list[dict], image_root: Path,
               cfg: TrainConfig, out_dir: Path, sample_prompts: list[str] = (), sample_seeds: list[int] = ()):
    """Fine-tune cross-attention LoRA adapters. Writes to out_dir:
    train_log.csv (every log_every steps), val_log.csv, samples/step_XXXXX.png,
    lora/ (weights + config) and summary.json (runtime, memory, instability counts)."""
    from diffusers import DDPMScheduler

    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    use_amp = c.device == "cuda"

    unet = c.unet
    lora_cfg = add_lora(unet, cfg.rank, cfg.alpha, cfg.target_modules)
    params = lora_parameters(unet)
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"Trainable LoRA parameters: {n_trainable / 1e6:.2f}M of {n_total / 1e6:.1f}M "
          f"({100 * n_trainable / n_total:.2f}%)")
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    noise_scheduler = DDPMScheduler.from_config(c.scheduler_config)  # the training forward process
    T = noise_scheduler.config.num_train_timesteps
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ds = CocoCaptionDataset(train_records, image_root, cfg.resolution, cfg.caption_dropout)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate,
                        num_workers=cfg.num_workers, generator=torch.Generator().manual_seed(cfg.seed),
                        persistent_workers=cfg.num_workers > 0)
    assert len(loader) > 0, "fewer training images than one batch"
    validation = FixedValidation(c, val_records, image_root, cfg.val_timesteps, noise_scheduler, cfg.resolution)

    train_log, val_log, events = [], [], []

    def run_val(step):
        unet.eval()
        row = {"step": step, **validation(c)}
        unet.train()
        val_log.append(row)
        print(f"  step {step:5d} | val loss {row['val_loss']:.4f}")

    def run_samples(step):
        if not sample_prompts:
            return
        unet.eval()
        res = generate(c, list(sample_prompts), list(sample_seeds), sampler="ddim",
                       num_inference_steps=cfg.sample_steps, guidance_scale=7.5)
        unet.train()
        w, h = res.images[0].size
        sheet = Image.new("RGB", (w * len(res.images), h))
        for k, im in enumerate(res.images):
            sheet.paste(im, (k * w, 0))
        sheet.resize((sheet.width // 2, sheet.height // 2)).save(out_dir / "samples" / f"step_{step:05d}.jpg", quality=90)

    run_val(0)
    run_samples(0)
    unet.train()
    reset_peak_memory(c.device)
    sync(c.device)
    t0 = time.perf_counter()
    step, micro, running = 0, 0, []
    window_start, window_steps = time.perf_counter(), 0

    while step < cfg.max_steps:
        for pixels, captions in loader:
            with torch.no_grad():
                z0 = encode_images(c, pixels)
                context = encode_prompt(c, captions)
            eps = torch.randn_like(z0)
            t = torch.randint(0, T, (z0.shape[0],), device=c.device)
            zt = noise_scheduler.add_noise(z0, eps, t)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = unet(zt, t, encoder_hidden_states=context).sample
            loss = F.mse_loss(pred.float(), eps.float())

            if not torch.isfinite(loss):
                events.append({"step": step, "event": "non-finite loss", "value": loss.item()})
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / cfg.grad_accum).backward()
            running.append(loss.item())
            micro += 1
            if micro % cfg.grad_accum:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                events.append({"step": step, "event": "fp16 overflow, step skipped", "grad_norm": grad_norm})
            optimizer.zero_grad(set_to_none=True)
            lr_sched.step()
            step += 1
            window_steps += 1

            if step % cfg.log_every == 0 or step == cfg.max_steps:
                sync(c.device)
                now = time.perf_counter()
                row = {"step": step, "loss": sum(running) / len(running), "lr": lr_sched.get_last_lr()[0],
                       "grad_norm": grad_norm, "sec_per_step": (now - window_start) / max(1, window_steps),
                       "elapsed_min": (now - t0) / 60, "peak_memory_mb": peak_memory_mb(c.device)}
                train_log.append(row)
                running, window_start, window_steps = [], now, 0
                if step % (cfg.log_every * 10) == 0:
                    print(f"  step {step:5d} | loss {row['loss']:.4f} | lr {row['lr']:.2e} | "
                          f"{row['sec_per_step']:.2f} s/step | {row['elapsed_min']:.1f} min")
            paused = time.perf_counter()
            if step % cfg.val_every == 0 or step == cfg.max_steps:
                run_val(step)
            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                run_samples(step)
            window_start += time.perf_counter() - paused  # keep validation and sampling out of s/step
            if step >= cfg.max_steps:
                break

    sync(c.device)
    train_minutes = (time.perf_counter() - t0) / 60
    if cfg.gradient_checkpointing:
        unet.disable_gradient_checkpointing()
    unet.eval()

    meta = {"rank": cfg.rank, "alpha": cfg.alpha, "target_modules": cfg.target_modules,
            "base_model": c.model_id, "steps": step, "trainable_params": n_trainable}
    save_lora(unet, out_dir / "lora", meta)
    _write_csv(out_dir / "train_log.csv", train_log)
    _write_csv(out_dir / "val_log.csv", val_log)
    summary = {
        "config": asdict(cfg),
        "lora": {**meta, "peft_config": {"r": lora_cfg.r, "lora_alpha": lora_cfg.lora_alpha}},
        "train_images": len(train_records),
        "val_images": len(val_records),
        "epochs": round(step * cfg.batch_size * cfg.grad_accum / max(1, len(train_records)), 2),
        "train_minutes": round(train_minutes, 2),
        "sec_per_step": round(60 * train_minutes / max(1, step), 3),
        "peak_memory_mb": peak_memory_mb(c.device),
        "val_loss_start": val_log[0]["val_loss"],
        "val_loss_end": val_log[-1]["val_loss"],
        "instability_events": events,
        "environment": environment_info(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Done: {step} steps in {train_minutes:.1f} min | val loss "
          f"{summary['val_loss_start']:.4f} -> {summary['val_loss_end']:.4f} | events: {len(events)}")
    return summary, train_log, val_log


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(f)]
