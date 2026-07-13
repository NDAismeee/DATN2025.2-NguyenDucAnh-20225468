# Bathymetry Experiments

This is a self-contained experiment pipeline. It does not dispatch to the old model folders.

## Data

Default Agia Napa paths:

- Images: `/mnt/disk3/anhnd2468/MagicBathyNet/agia_napa/img/aerial`
- Depth: `/mnt/disk3/anhnd2468/MagicBathyNet/agia_napa/depth/aerial`

## Models

Supported model keys: `proposed`, `cnn`, `knn`, `depth_anything_v2`, `unet`, `random_forest`, `da_sdb`, `dpt`, and `mlp`.

`da_sdb` is implemented as a local domain-adaptive SDB-style neural baseline with a shallow-water prior branch because the external paper implementation is not included in this repository. `dpt` is a compact Dense Prediction Transformer implemented locally for reproducible training without downloading external weights.

## Commands

```bash
export PYTHONPATH=src
python -m bathymetry_experiments.cli train --model random_forest --config configs/agia_napa.yaml
python -m bathymetry_experiments.cli experiment --models knn random_forest mlp cnn unet proposed da_sdb dpt depth_anything_v2 --config configs/agia_napa.yaml
python -m bathymetry_experiments.cli scatter --pred-dir runs/random_forest/<run>/infer_outputs
```

Each run writes `config_used.yaml`, `metrics.csv`, `predictions.csv`, `summary.json`, model weights, and `infer_outputs` with `*_pred.npy`, `*_gt.npy`, `*_valid_mask.npy`, and `infer_summary.csv`.

## Verification

```bash
python -m compileall .
PYTHONPATH=src python -m bathymetry_experiments.cli --help
PYTHONPATH=src python -m bathymetry_experiments.cli train --model knn --config configs/agia_napa.yaml
PYTHONPATH=src python -m bathymetry_experiments.cli scatter --pred-dir <infer_outputs>
```
