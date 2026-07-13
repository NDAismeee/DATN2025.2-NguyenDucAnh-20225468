# LLM-Guided Bathymetry

Reference implementation of the VLM-Guided Shallow-Water Depth Estimation framework
described in `main.pdf` (Sections 4.1–4.8).

## Pipeline overview

1. Query a pretrained VLM once per aerial RGB tile to obtain disturbance regions,
   textual descriptions, depth-prior intervals, and confidences.
2. Rasterize the VLM JSON into per-pixel maps:
   `unreliable_mask`, `region_masks`, `d_phys`, `d_min`, `d_max`, `gamma_map`, `w_phys`.
3. Encode region descriptions to fixed-dim text embeddings.
4. Train an end-to-end model that fuses noise-aware visual features with semantic
   expert knowledge and applies a confidence- and interval-aware soft gate to mix
   the neural depth prediction with the physical prior.

## Scene format

Each scene folder (or sidecar folder for paired-aerial layout) should contain:

- `image.png` or `image.tif`
- `depth.npy` or `depth.tif`
- `valid_mask.npy` (optional; inferred from depth if absent)
- `llm_output.json`
- generated `unreliable_mask.npy`, `region_masks.npy`, `d_phys.npy`, `d_min.npy`,
  `d_max.npy`, `gamma_map.npy`, `w_phys.npy`, `text_embeddings.npy`

## Allowed VLM categories

`sun_glint`, `shadow`, `turbidity`, `foam`, `ambiguous_bottom`, `other`
(matches Section 4.2 of the paper).

Each disturbance region must include a `failure_direction` from
`{artificially_shallow, artificially_deep, ambiguous}`.

## Commands

```bash
python llm_pipeline/query_llm.py --scene data/scenes/agia_napa_001
python llm_pipeline/build_prior_maps.py --scene data/scenes/agia_napa_001
python llm_pipeline/encode_text.py --scene data/scenes/agia_napa_001
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --domain_min 0 --domain_max 30
python scripts/predict.py --scene data/scenes/agia_napa_001 --checkpoint outputs/checkpoints/best.pt
python scripts/visualize.py --scene data/scenes/agia_napa_001
python scripts/validate_vlm_priors.py --config configs/agia_napa_aerial_train.yaml --reference-mask-dir <ref>
```

## Defaults aligned with paper (Sec 5.1)

| Setting | Value |
|---|---|
| Patch size | 720 × 720 |
| Hidden dim | 256 |
| Gate MLP | 128 units, dropout 0.1 |
| Variance lower bound | 1e-4 |
| Optimizer | AdamW lr=1e-4, wd=1e-4 |
| LR schedule | 5-epoch linear warm-up → cosine decay |
| Gradient clip | 1.0 |
| Epochs | 100, early stop patience 15 (on val MAE) |
| `λ_align`, `λ_int` | 1e-2, 1e-2 |
| Contrastive temperature | 0.07 |

The VLM/LLM is queried offline. The neural model does not train the language model
or the text encoder.
