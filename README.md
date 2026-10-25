# IE7615 Group 3 — Project 2: Text-to-Image Generation

A text-to-image pipeline built on a Transformer text encoder and a diffusion-based image generator, using a curated subset of COCO Captions.

## Milestone 1 summary

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
