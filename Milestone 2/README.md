# Milestone 2 — Model Integration and Baseline Results

The CLIP text encoder and the Stable Diffusion 1.5 diffusion model are wired together by hand and verified, and the integrated pipeline produces the baseline conditional outputs.

## Status

| Milestone 2 task | Where | Status |
|---|---|---|
| 1. Integrate the text encoder with a diffusion model | `01_integration_and_training.ipynb` | Drafted |
| 2. Run baseline conditional generation | `02_conditional_generation_sweeps.ipynb` | Drafted |
| 3. Guidance-scale experiments | `02_conditional_generation_sweeps.ipynb`, Section 3 | Drafted |
| 4. Noise schedule and sampling experiments | `02_conditional_generation_sweeps.ipynb`, Sections 4 and 5 | Drafted |
| 5. Training logs and summary | to add to notebook 1 (see its Section 8), plus `docs/training_logs_summary.md` | Not started |

Each notebook ends with a "Next steps" section listing the cells to add for the remaining tasks and the existing helpers in `t2i.py` and `lora.py` they can use.

## Files

| Path | Description |
|---|---|
| `t2i.py` | The integrated pipeline: loads each SD 1.5 component, encodes prompts, runs the guided denoising loop with any sampler, records cross-attention maps and sampler diagnostics, scores images with CLIP ViT-B/32, and saves every image with its metadata |
| `lora.py` | LoRA fine-tuning of the UNet's cross-attention on the Milestone 1 COCO subset. Not used by the notebooks yet; it is ready for the "modify the conditioning path" and training-log steps |
| `01_integration_and_training.ipynb` | Integration notebook: component tour, shape trace, equivalence check against `diffusers`, conditioning ablations, cross-attention maps |
| `02_conditional_generation_sweeps.ipynb` | Generation notebook: 20 baseline prompts × 3 seeds, CLIP alignment against the real COCO photos, labeled baseline grids; guidance-scale, sampler, step-count and noise-schedule sweeps on 6 prompts; samples gallery PDF |
| `docs/integration_writeup.md` | Integration write-up draft (export to PDF for Canvas). Its LoRA parts describe the planned fine-tune |
| `docs/training_logs_summary.md` | Training logs and summary template, for Task 5 |
| `docs/architecture.svg` | Architecture diagram; the dashed branch is the planned fine-tune |
| `outputs/` | Created by the notebooks: `generated/baseline_base/` (images + JSON + `manifest.jsonl`), `figures/`, `logs/` |

## Running

Use a CUDA GPU (a Colab T4 is enough). Install the root `requirements.txt`, then run:

1. **`01_integration_and_training.ipynb`**
2. **`02_conditional_generation_sweeps.ipynb`** (60 baseline images, then about 280 sweep images)

Notebook 2 resumes: an image with an existing JSON file is not regenerated. The first run downloads SD 1.5 (about 4 GB) and CLIP ViT-B/32 (about 600 MB) into the Hugging Face cache.

**TODO (team):** measured run times.

## Reproducibility

- Seeds are the Milestone 1 `generation_seed` values (1001 to 1020) plus fixed offsets (0, 100, 200). Each image gets its own CPU generator, so results do not depend on batch size or GPU.
- The hand-written loop reproduces `StableDiffusionPipeline` exactly with the same components and seeds (notebook 1, Section 4).
