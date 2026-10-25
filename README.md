# IE7615 Group 3 — Project 2: Text-to-Image Generation

A text-to-image generative pipeline built on a Transformer text encoder and a diffusion-based image generator, trained and evaluated on a curated subset of COCO Captions.

## Repository contents

| Path | Description |
|---|---|
| `Milestone 1/DatasetPreparation.ipynb` | Downloads a 3,000-image subset of COCO 2017, cleans and deduplicates captions, preprocesses images, and exports the curated dataset |
| `Milestone 1/clip_embeddings.ipynb` | Loads a pretrained CLIP model, tokenizes captions, computes image and text embeddings, and checks that matching image–caption pairs score highest |
| `Milestone 1/coco_export/labels.json` | Curated image–caption pairs: each record holds an image path, its cleaned captions, and its COCO image ID, file name, and license |
| `requirements.txt` | Pinned Python packages for every notebook in the project |

Large files (the COCO caption annotations, exported images, checkpoints, and embeddings) are not stored in the repository. Each notebook recreates what it needs.

## Environment setup

Requires Python 3.9 to 3.12. Python 3.13 is not supported by the pinned package versions.

### Local (macOS or Linux)

```bash
git clone https://github.com/rasmussenj03/IE7615-Group3-Project2.git
cd IE7615-Group3-Project2
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ie7615-project2
```

### Local (Windows)

```bash
git clone https://github.com/rasmussenj03/IE7615-Group3-Project2.git
cd IE7615-Group3-Project2
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ie7615-project2
```

When opening a notebook, select the `ie7615-project2` kernel.

### Google Colab

1. Add a Colab secret named `GITHUB_TOKEN` holding a GitHub access token with read access to this repository.
2. Select a GPU runtime under Runtime → Change runtime type.
3. Open `Milestone 1/clip_embeddings.ipynb` and run its first cell. It mounts Google Drive, clones the repository into `MyDrive/IE7615-Group3-Project2` on the first run, and installs `requirements.txt`. If Colab asks to restart the runtime after the install, restart and run the cell again.

To clone into the temporary Colab session instead of Drive, set `COLAB_USE_DRIVE = False` in that cell. Files saved there are lost when the runtime ends.

### Verify the environment

```bash
python -c "import torch, transformers, diffusers, fiftyone; print(torch.__version__)"
```

This should print `2.8.0`. The CLIP notebook uses a CUDA GPU when available, an Apple Silicon GPU (MPS) on Mac, and the CPU otherwise.

## Reproducing the results

Run the notebooks from inside `Milestone 1/`, which is where Jupyter starts when you open them there. `clip_embeddings.ipynb` also switches into that folder on its own. Run them in this order:

1. **`DatasetPreparation.ipynb`** — downloads 3,000 COCO images and the caption annotations into `Milestone 1/dataset/`, then builds the curated dataset in `Milestone 1/coco_export/`.
2. **`clip_embeddings.ipynb`** — computes CLIP embeddings for a sample of 256 pairs and saves them to `Milestone 1/outputs/clip/clip_embeddings.pt`.

`clip_embeddings.ipynb` samples with a fixed seed (42), so every run uses the same pairs.

## Dataset

COCO Captions 2017 (train split), accessed through FiftyOne. COCO annotations are released under CC BY 4.0; individual images carry their own Flickr licenses.
