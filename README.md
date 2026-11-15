# IE7615 Group 3 — Project 2: Text-to-Image Generation

A text-to-image pipeline built on a Transformer text encoder and a diffusion-based image generator, using a curated subset of COCO Captions.

## Milestone 1 Summary

| Notebook | What it does | Result |
|---|---|---|
| `Milestone 1/DatasetPreparation.ipynb` | Downloads 3,000 COCO 2017 train images with FiftyOne, cleans and deduplicates captions, resizes and center-crops images to 512×512, and exports the curated dataset | 3,000 image–caption pairs in `coco_export/` |
| `Milestone 1/clip_embeddings.ipynb` | Loads pretrained CLIP (ViT-B/32), tokenizes captions, and computes image and text embeddings for 256 sampled pairs | Image→text R@1 0.69, R@5 0.96; text→image R@1 0.65, R@5 0.94 (chance: 0.004) |
| `Milestone 1/baseline_generation.ipynb` | Generates images for 20 dataset captions with pretrained Stable Diffusion 1.5 (25 steps, guidance 7.5, 512×512), compares each with its original COCO image, and includes the qualitative memo | 20/20 images and side-by-side comparisons |

`coco_export/labels.json` holds each image's path, cleaned captions, and COCO image ID, file name, and license. Images, downloads, checkpoints, and embeddings are not stored in the repository; the notebooks recreate them.

## Setup

Requires Python 3.9–3.12.

```bash
git clone https://github.com/rasmussenj03/IE7615-Group3-Project2.git
cd IE7615-Group3-Project2
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open the notebooks with the `ie7615-project2` kernel.

**Google Colab:** select a GPU runtime, then run the first cell of `DatasetPreparation.ipynb` or `clip_embeddings.ipynb`. It clones the repository and installs `requirements.txt`.

## Running Milestone 1

Run the notebooks in this order:

1. `DatasetPreparation.ipynb` — builds `coco_export/`
2. `clip_embeddings.ipynb` — saves embeddings to `outputs/clip/`
3. `baseline_generation.ipynb` — saves generated images and comparisons to `Project2_Baseline/outputs/`

All sampling uses seed 42, so reruns select the same images and prompts.

## Dataset

COCO Captions 2017 (train split). Annotations are licensed under CC BY 4.0; each image carries its own Flickr license, recorded in `labels.json`.

# Milestone 2 Summary

The CLIP text encoder and the Stable Diffusion 1.5 diffusion model are wired together by hand and verified, and the integrated pipeline produces the baseline conditional outputs.

| Notebook | What it does | Result |
|---|---|---|
| `01_integration_and_training.ipynb` |  Integrate the text encoder with a diffusion model | Outputs go to `Milestone 2/outputs/logs` and `Milestone 2/outputs/figures` |
| `02_conditional_generation_sweeps.ipynb` | Run baseline conditional generation | Outputs go to `Milestone 2/outputs/generated`, with the `manifest.jsonl` file holding all generation information. |
| `02_conditional_generation_sweeps.ipynb`, Section 3 | Guidance-scale experiments | Summary available in `outputs/gallery_base.pdf` |
| `02_conditional_generation_sweeps.ipynb`, Sections 4 and 5 | Noise schedule and sampling experiments | Summary available in `outputs/gallery_base.pdf` |
| to add to notebook 1 (see its Section 8), plus `docs/training_logs_summary.md` | Training logs and summary | Summary available in `outputs/gallery_base.pdf` |

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

**Measured Run Time and Compute**
| Experiment | Images | s / image | Attrib. min | Peak MiB |
|---|---|---|---|---|
| Baseline | 60 | 6.63 | 6.63 | 5080.9 | 
| Guidance sweep | 126 | 6.26 | 13.15 | 5080.2 |
| Sampler / step sweep | 126 | 3.78 | 7.94 | 5080.5 |
| 100-step reference | 6 | 13.43 | 1.34 | 5080.4 |
| Timestep spacing | 36 | 2.22 | 1.33 | 5080.2|
| Schedule mismatch | 18 | 6.80 | 2.04 | 5080.2 |

## Reproducibility

- Seeds are the Milestone 1 `generation_seed` values (1001 to 1020) plus fixed offsets (0, 100, 200). Each image gets its own CPU generator, so results do not depend on batch size or GPU.
- The hand-written loop reproduces `StableDiffusionPipeline` exactly with the same components and seeds (notebook 1, Section 4).

