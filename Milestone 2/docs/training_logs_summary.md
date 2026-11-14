# Milestone 2 — Training Logs and Summary

**IE7615 Group 3 · Project 2** · DRAFT. Fill the **TODO** values from `Milestone 2/outputs/logs/` and `outputs/lora_run/summary.json` after the full run. Every figure referenced here is in `outputs/figures/` and in the gallery PDF.

## 1. Runs and environment

| Run | Notebook | Hardware | Wall time | Peak GPU memory |
|---|---|---|---|---|
| Component load + integration checks | 01, §2–8 | **TODO** (e.g. Colab T4 16 GB) | **TODO** min | **TODO** MB |
| Cross-attention LoRA fine-tune, 1,500 steps | 01, §9 | **TODO** | **TODO** min (**TODO** s/step) | **TODO** MB |
| Baseline: 20 prompts × 3 seeds | 02, §2 | **TODO** | **TODO** min | **TODO** MB |
| Guidance sweep: 126 images | 02, §3 | **TODO** | **TODO** min | **TODO** MB |
| Sampler × steps sweep: 132 images | 02, §4 | **TODO** | **TODO** min | **TODO** MB |
| Schedule experiments: 60 images | 02, §5 | **TODO** | **TODO** min | **TODO** MB |

Package versions are in `outputs/logs/environment.json` and in every image's JSON file: torch 2.8.0, diffusers 0.36.0, transformers 4.57.6, peft 0.17.1. Inference runs in fp16 at 512×512.

## 2. Fine-tuning logs (cross-attention LoRA)

Setup:
- rank 8, α = 8, about 0.80M trainable parameters; AdamW at lr 1e-4 with 100 warmup steps, then cosine decay to 1e-5;
- batch 4, 1,500 steps ≈ **TODO** epochs over **TODO** training images, with 10% caption dropout;
- 20 held-out images for validation.

![Loss curves](../outputs/figures/08_lora_loss_curves.png)

- **Training loss** (ε-MSE, logged as a 10-step mean plus EMA) is noisy because each batch draws random timesteps. Low-t steps have tiny loss and high-t steps large loss.
- **Validation loss** uses fixed timesteps {100, 300, 500, 700, 900} and fixed noise on the held-out images: **TODO** → **TODO**. The per-timestep panel shows where the fine-tune helped (**TODO**: expected mid-range t, where the caption matters most).
- **Stability:** fp16 overflow or skipped steps: **TODO** (list from `summary.json → instability_events`). Gradient norm stayed below the clip value of 1.0: **TODO**.
- **Samples during training** (`09_lora_training_samples.png`): **TODO**, for example a drift toward a COCO-snapshot look by step 500.

## 3. Sampler diagnostics (inference)

![Sampler diagnostics](../outputs/figures/05_sampler_diagnostics.png)

- **Latent std** falls from about 1 to the scale of clean latents over the run. DDPM is noisier step to step because it injects fresh noise.
- **‖ε_c − ε∅‖** (how strongly the text steers each step) is **TODO**: expected largest early to mid trajectory, when layout is decided, and small near t = 0, when only texture is refined.
- **x̂₀ previews** (`06_x0_previews.png`): the composition is visible by about step **TODO** of 50, and later steps only add detail. This is why few-step solvers still get the layout right.

## 4. Parameter-tuning observations

**Guidance scale** (DDIM 50; 6 prompts × 3 seeds; `30_guidance_*`):

| w | 0 | 1 | 3 | 5 | 7.5 | 10 | 15 |
|---|---|---|---|---|---|---|---|
| CLIP score | **TODO** | | | | | | |
| seed diversity | **TODO** | | | | | | |
| saturation | **TODO** | | | | | | |

**TODO** observations: alignment rises steeply from w = 1 to w ≈ 5 and then plateaus; diversity falls; above w ≈ 10, colors over-saturate and textures look "fried"; w = 1 costs half as much compute. Our pick for Milestone 3: **TODO**.

**Sampler and step count** (w = 7.5; `40_samplers_*`):
- **Quality vs compute:** **TODO**, the cheapest setting within about 1 dB of the 100-step reference.
- **DDPM at 10 steps:** **TODO** (expected to fail, because its posterior variance assumes small steps).
- **DPM++ 2M Karras at 10 to 20 steps:** **TODO** (expected near-converged).
- Time per image scales linearly with steps (**TODO** s per step at batch 4).

**Noise schedule** (`50_spacing_*`, `51_schedule_mismatch_*`):
- **Timestep spacing at 10 steps:** `leading` vs `trailing` gives **TODO**.
- **Karras σ:** **TODO**.
- **Mismatched β schedule:** **TODO**. This confirms the schedule is baked into the weights.

## 5. What worked, what did not, what we would change

**Worked**
- The hand-wired pipeline matched the reference implementation exactly. Seeded per-image generators made every image reproducible across batch sizes.
- Resumable, metadata-rich runs (one JSON per image plus a manifest) made the sweeps easy to restart after Colab disconnects and easy to reuse in Milestone 3.
- **TODO**

**Did not work or was harder than expected**
- **TODO**, for example compositional prompts (prompt 14: dog *with a red frisbee*), counting (prompt 10: three people), and two-deck buses. These are attribute-binding failures of the CLIP-conditioned UNet that guidance alone does not fix.
- **TODO**, for example the fine-tune shifting style more than alignment.

**Next (Milestone 3)**
- FID and Inception Score on the baseline set, plus CLIP-score alignment with CLIP ViT-L/14 as a second judge.
- Sensitivity study over guidance × steps × sampler, re-using the saved sweep outputs, with more seeds for confidence intervals.
- An embedding-choice dimension: base vs LoRA context, and optionally prompt-embedding interpolation.
- If time allows, a small 32×32 from-scratch conditional DDPM to compare linear vs cosine training schedules directly.
