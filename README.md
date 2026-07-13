# DATN2025.2-NguyenDucAnh-20225468

Source code for the graduation thesis **RGB-Based Shallow-Water Bathymetry with LLM-Derived Priors**.

The project studies dense shallow-water bathymetry estimation from high-resolution aerial RGB imagery. The main proposed method, **BathyAgent**, combines a neural dense-depth predictor with offline guidance from a multimodal LLM/VLM. The LLM identifies visually unreliable water regions and returns structured annotations containing polygons, disturbance descriptions, plausible local depth intervals, confidence scores, and rationales. These annotations are converted into raster masks, text embeddings, confidence maps, and local numerical priors, then fused with visual features through semantic grounding and a confidence-constrained adaptive gate.

## Main Ideas

- **Input**: aerial RGB image patches and rasterized bathymetry labels from MAGICBATHYNET.
- **Task**: predict dense pixel-wise water depth over valid water pixels.
- **Challenge**: RGB appearance is ambiguous under sun glint, shadows, turbidity, foam, heterogeneous seabed material, and weak bottom visibility.
- **Proposed solution**: use LLM-derived region-level priors only where visual evidence is unreliable, while preserving neural prediction elsewhere.
- **Metrics**: MAE, RMSE, and standard deviation of absolute errors, computed over valid water pixels.

## Reported Results

The thesis evaluates Agia Napa and Puck Lagoon aerial RGB subsets. In the reported within-site experiments, DPT gives the best overall metric accuracy, while BathyAgent consistently ranks second and improves robustness in ambiguous/deep-water regions.

| Method | Agia Napa MAE | Agia Napa RMSE | Puck Lagoon MAE | Puck Lagoon RMSE |
|---|---:|---:|---:|---:|
| DPT | 0.238 | 0.271 | 0.157 | 0.221 |
| BathyAgent | 0.296 | 0.385 | 0.169 | 0.229 |
| Depth Anything V2 | 0.319 | 0.415 | 0.176 | 0.236 |
| U-Net | 0.420 | 0.616 | 0.187 | 0.298 |

The ablation study shows that the LLM-derived local depth prior, problem-region mask, adaptive soft gate, and region-text alignment loss all contribute to the final model. Cross-site experiments also show BathyAgent outperforming DA-SDB and Depth Anything V2 when transferring between Agia Napa and Puck Lagoon.

## Repository Structure

```text
.
├── README.md
└── source_code/
    ├── bathyagent/                 # Main BathyAgent implementation
    ├── bathymetry_experiments/     # Unified benchmark pipeline for baselines and proposed variants
    ├── bathymetry_llm/             # Earlier/reference LLM-guided bathymetry pipeline
    ├── cnn/                        # Legacy CNN experiment code
    ├── cnn_src/                    # CNN training/inference scripts
    ├── da-sdb/                     # DA-SDB baseline code
    ├── depth_anythingv2/           # Depth Anything V2 baseline code
    ├── dpt/                        # DPT baseline code
    ├── mlp/                        # MLP baseline code
    ├── rf/                         # Random Forest baseline code
    └── unet/                       # U-Net baseline code
```

The recommended entry points are `source_code/bathymetry_llm` for the thesis reference pipeline, `source_code\bathyagent` for the current BathyAgent annotation/training workflow, and `source_code\bathymetry_experiments` for a compact unified comparison pipeline.

## Data

The code expects MAGICBATHYNET-style paired aerial RGB and bathymetry rasters. The default configs use environment variables so local data paths do not need to be hard-coded:

```text
IMAGE_DIR=/path/to/MagicBathyNet/agia_napa/img/aerial
DEPTH_DIR=/path/to/MagicBathyNet/agia_napa/depth/aerial
SEMANTIC_DIR=/path/to/generated/semantic_maps
ANNOTATION_JSON_DIR=/path/to/generated/annotations
OPENAI_API_KEY=your_api_key_here
```

For `bathyagent`, copy `source_code/bathyagent/.env.example` to `source_code/bathyagent/.env` and update the paths. Depth files are matched against image stems using suffixes such as `_depth`, `_bathy`, `_gt`, and `_label`.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the main BathyAgent dependencies:

```powershell
python -m pip install -r source_code\bathyagent\requirements.txt
```

For the unified experiment pipeline:

```powershell
python -m pip install -r source_code\bathymetry_experiments\requirements.txt
python -m pip install -e source_code\bathymetry_experiments
```

If CUDA is available, install a PyTorch build compatible with the local CUDA driver before running GPU experiments.

## Thesis Reference Pipeline

Run these commands from `source_code/bathymetry_llm`.

```powershell
python llm_pipeline\query_llm.py --scene data\scenes\<scene_id>
python llm_pipeline\build_prior_maps.py --scene data\scenes\<scene_id>
python llm_pipeline\encode_text.py --scene data\scenes\<scene_id>
python scripts\train.py --config configs\agia_napa_aerial_train.yaml
python scripts\evaluate.py --config configs\default.yaml --checkpoint outputs\checkpoints\best.pt
python scripts\predict.py --scene data\scenes\<scene_id> --checkpoint outputs\checkpoints\best.pt
```

This path represents the paper-style workflow: query the VLM offline, build prior maps, encode region text, then train/evaluate the depth model.

## BathyAgent Workflow

Run these commands from `source_code/bathyagent`.

1. Generate LLM/VLM annotations:

```powershell
python llm_annotate.py --config config.yaml --limit 10
```

2. Rasterize annotations into semantic maps and local priors:

```powershell
python rasterize_semantics.py --config config.yaml
```

3. Precompute text embeddings:

```powershell
python precompute_text_embeddings.py --config config.yaml
```

4. Train and evaluate BathyAgent:

```powershell
python main.py --config config.yaml --train 1 --test 1 --device cuda
```

5. Run inference and visualization:

```powershell
python infer.py --config config.yaml --checkpoint logs\<run_name>\<run_id>\best_model.pt --sample_idx 0
python infer.py --config config.yaml --checkpoint logs\<run_name>\<run_id>\best_model.pt --all --no_show --output_dir infer_outputs
```

Main generated outputs include `train_log.csv`, `metrics.csv`, `predictions.csv`, `best_model.pt`, `last_model.pt`, per-sample `.npy` predictions, and visualization PNGs.

## Unified Experiment Pipeline

Run these commands from `source_code/bathymetry_experiments`.

```powershell
$env:PYTHONPATH = "src"
python -m bathymetry_experiments.cli train --model random_forest --config configs/agia_napa.yaml
python -m bathymetry_experiments.cli experiment --models knn random_forest mlp cnn unet proposed da_sdb dpt depth_anything_v2 --config configs/agia_napa.yaml
python -m bathymetry_experiments.cli scatter --pred-dir runs\random_forest\<run_id>\infer_outputs
```

Supported model keys are:

```text
proposed, cnn, knn, depth_anything_v2, unet, random_forest, da_sdb, dpt, mlp
```

Each run writes `config_used.yaml`, `metrics.csv`, `predictions.csv`, `summary.json`, model weights, and `infer_outputs`.

## Configuration Notes

- `source_code/bathyagent/config.yaml` contains the main BathyAgent architecture, training, data, semantic-map, LLM, text-encoder, and logging settings.
- `source_code/bathymetry_llm/configs/*.yaml` contains the thesis reference pipeline settings.
- `source_code/bathymetry_experiments/configs/*.yaml` contains compact configs for baseline and proposed-method experiments.
- Default dataset paths in sample configs are placeholders or machine-specific paths. Update them before running.
- LLM annotations are generated offline. The neural model trains on saved annotations, raster maps, and text embeddings; it does not train the LLM or text encoder.

## Generated Files

The repository ignores generated experiment artefacts by default, including local environments, caches, logs, model checkpoints, `.env` files, annotation outputs, semantic maps, inference outputs, and large raster/data files. Keep raw datasets and trained weights outside git unless there is a specific reason to version a small sample.