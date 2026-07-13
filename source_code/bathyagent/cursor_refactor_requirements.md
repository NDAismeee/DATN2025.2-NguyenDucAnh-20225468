# Refactor Requirements for Cursor

Please refactor the current codebase so it fully matches the intended simplified design below.

## Target design

The final system must use:

- **RGB image input**
- **one single semantic channel**: `issue_map`
- **OpenAI API via `OPENAI_API_KEY`** for offline annotation
- **no coastal zone channels** such as:
  - `nearshore`
  - `transition`
  - `offshore`
- **no per-zone depth ranges**
- **no old local Qwen path as the default annotation backend**
- **no raster prior / LLM prior in the default training path**

The default active pipeline after refactor must be:

- encoder input = **4 channels total**
  - 3 RGB channels
  - 1 `issue_map` channel
- auxiliary tensors may still exist outside the encoder:
  - `water_mask`
  - `valid_mask`
  - `depth_gt`

---

## 1. Fix `config.yaml`

Update the config so it reflects the new actual pipeline.

### Required changes

- set semantic channels to exactly one channel:
  - `semantic.channel_order: ["issue_map"]`
- keep:
  - `data.image_mode: rgb`
  - `data.selected_bands: null`
  - `model.fuse_semantic_channels: true`
- disable prior-based logic by default:
  - `semantic.use_prior_depth_map: false`
  - `model.use_raster_prior_depth: false`
  - `model.use_llm_prior: false`
  - `model.use_real_llm: false`
- disable old prior-fusion defaults:
  - `model.use_dual_uncertainty: false` by default unless explicitly needed only for the neural head
- remove or ignore old dependencies on:
  - `risk_score`
  - `llm_depth_source`
  - `risk_score_channel_name`

### Expected result

The config should represent a clean **RGB + issue_map** training setup with no hidden dependence on the old 12-channel semantic stack.

---

## 2. Fix `common.py`

### Required changes

Update `sync_model_encoder_layout()` so that under the default config:

- `image_channels = 3`
- `semantic_extra_channels = 1`
- `encoder_in_channels = 4`

Update validation and pipeline print statements so they clearly describe:

- default pipeline = **RGB + single issue_map**
- offline annotation backend = **OpenAI API**
- old 12-channel / zone-based setup is no longer the default path

### Important

Do not let the logs or validation text imply that the old 12-channel zone/risk stack is still the main design.

---

## 3. Fix `dataset.py`

### Required changes

Ensure the dataset correctly supports the simplified semantic stack.

### Must support

- `DEFAULT_SEMANTIC_CHANNEL_ORDER = ["issue_map"]`
- loading semantic file as one channel:
  - if semantic file shape is `(H, W)`, convert to `(1, H, W)`
  - if already `(1, H, W)`, keep it
- `water_mask` may still be loaded separately if available
- `prior_depth_map` must become optional and unused by default

### Important

The dataset must not assume the old 12 semantic channels are still standard.

The active/default path must work correctly when semantic input contains only one channel.

---

## 4. Fix `llm_annotate.py`

### Required changes

Refactor the offline annotation script to use the simplified schema and OpenAI backend.

### Replace old behavior

Do **not** ask the model for:

- shoreline
- `nearshore`
- `transition`
- `offshore`
- zone depth ranges
- coastal zone partitioning

### New annotation output schema

The JSON must contain only:

```json
{
  "water": {
    "polygons": [[[x, y], [x, y], [x, y]]]
  },
  "uncertainty_regions": [
    {
      "polygons": [[[x, y], [x, y], [x, y]]],
      "issue_type": "turbidity",
      "risk_level": "high",
      "description": "...",
      "model_hint": "..."
    }
  ]
}
```

### New prompt requirements

The annotation prompt must request only:

1. water polygons
2. localized uncertainty regions inside water

Allowed `issue_type` values:

- `sun_glint`
- `turbidity`
- `shadow`
- `wave_roughness`
- `bottom_confusion`
- `color_ambiguity`
- `sensor_artifact`

### Important

- no zones
- no depth ranges
- no shoreline output
- no old schema normalization for zone-based outputs

---

## 5. Fix `vlm_utils.py`

### Required changes

Use **OpenAI API** as the default annotation backend.

### Must support

- `OPENAI_API_KEY`
- OpenAI Python SDK
- configurable model name from config, e.g. `gpt-4o`
- optional `base_url`
- existing helper ideas may remain:
  - `GenerationConfig`
  - JSON extraction
  - schema validation

### Must not be the default anymore

- local Qwen / transformers VLM loading
- `QwenVLModel`
- local `AutoProcessor` + `Qwen2_5_VLForConditionalGeneration` as the main annotation path

### Optional

Legacy local model code may remain only if clearly separated as a disabled fallback path. It must not be the default or assumed workflow.

---

## 6. Fix `rasterize_semantics.py`

### Required changes

Remove the old zone-based rasterization logic from the active path.

### Remove old active logic

Do not keep active logic for:

- `nearshore`
- `transition`
- `offshore`
- zone fallback partitioning
- exclusive zone enforcement
- prior depth generation from zones
- zone depth ranges

### New outputs

This script should now produce:

- `semantic_channels.npy` containing exactly one channel:
  - `issue_map`
- optionally:
  - `water_mask.npy`

### New `issue_map` logic

Merge all `uncertainty_regions` into a single semantic band.

Two acceptable implementations:

#### Option A: binary issue map
- `1.0` if the pixel belongs to any uncertainty region
- `0.0` otherwise

#### Option B: soft issue map
Map risk levels to values:
- `low -> 0.33`
- `medium -> 0.66`
- `high -> 1.0`

Use max pooling over overlapping regions.

### Important

`issue_map` should represent **all problematic regions merged into one band**.

No old multi-channel semantic stack should be emitted in the default mode.

---

## 7. Fix `model.py`

### Required changes

Make the default model path match the simplified architecture.

### Default intended model behavior

- input to encoder = RGB + `issue_map`
- model predicts depth directly from this 4-channel encoder input
- optional reconstruction branch may stay
- optional neural uncertainty head may stay

### Must not be active by default

- zone-based prior construction
- per-zone depth range inference
- `LocalQwenPrior`
- old local online Qwen logic
- raster prior fusion
- `depth_llm` / `d_phys` pathway as the default training behavior
- contrastive alignment against a prior branch
- dual-uncertainty fusion with a non-existent prior branch

### Important

If old prior branch code is kept for future experiments, it must be:

- fully optional
- behind clear flags
- not assumed in the default forward path
- not dependent on zone channels that no longer exist

### Expected default training path

A clean path like:

- concat RGB + issue_map
- encode
- predict depth
- compute masked depth loss
- optionally compute reconstruction loss
- return simple outputs

---

## 8. Fix `main.py`

### Required changes

Simplify training and logging so they align with the new default path.

### Keep

- standard training loop
- metrics
- checkpointing
- CSV logging

### Change

Make prior-related tensors truly optional:

- `prior_depth_map` should not be required in the default path
- prior-related diagnostics must not be assumed to exist

### Remove or guard conditionally

Do not assume the following always exist:

- `depth_llm`
- `w_llm`
- `var_llm`
- `risk_score`
- `alpha`
- prior branch losses

Only log those if a prior branch is explicitly enabled in the future.

### Expected default behavior

The normal run should work cleanly with:

- `image`
- `semantic_channels`
- `depth`
- `valid_mask`
- optional `water_mask`

and no prior branch.

---

## 9. Fix `viz_llm_predictions.py`

### Required changes

Update visualization so it reflects the simplified annotation/rasterization outputs.

### It should visualize

- RGB image
- water mask or water polygons
- uncertainty regions
- merged `issue_map`

### It should not assume

- old 12-channel semantic stack
- nearshore / transition / offshore outputs
- prior depth map as a required artifact

---

## 10. Fix `prompt_llm_depth_viz.py`

### Required changes

If this script is kept, make it clearly an optional analysis/debug tool only.

### It may do

- ask OpenAI for a coarse depth grid for inspection
- render that result for debugging

### It must not do

- silently reintroduce the old zone-based prior system
- be treated as the default training prior branch

---

## 11. Final expected architecture

After the refactor, the final active/default system must satisfy all of the following:

### Encoder input
Exactly **4 channels total**:

1. `R`
2. `G`
3. `B`
4. `issue_map`

### Auxiliary tensors allowed outside encoder
These may still exist:

- `water_mask`
- `valid_mask`
- `depth_gt`

These do **not** count as encoder channels.

### Annotation backend
Default annotation backend must use:

- `OPENAI_API_KEY`
- OpenAI Python SDK
- model name from config, e.g. `gpt-4o`

### Semantic representation
Only one semantic channel in the active/default pipeline:

- `issue_map`

No active default use of:

- `water` as semantic encoder channel
- `nearshore`
- `transition`
- `offshore`
- separate per-issue semantic channels
- `risk_score`

---

## 12. Recommended implementation order

Please apply changes in this order:

### Step 1
Fix `config.yaml` first so the intended mode is explicit.

### Step 2
Refactor `vlm_utils.py` and `llm_annotate.py` to OpenAI API + simplified schema.

### Step 3
Refactor `rasterize_semantics.py` to emit only:
- `issue_map`
- optional `water_mask`

### Step 4
Refactor `dataset.py` and `common.py` so the 4-channel encoder setup is guaranteed.

### Step 5
Refactor `model.py` to make the default path a clean RGB + issue_map model.

### Step 6
Refactor `main.py` and the visualization scripts so they no longer assume the old prior-based outputs.

---

## 13. Final completion checklist

The refactor is complete only if all of the following are true:

- [ ] `config.yaml` uses `channel_order: ["issue_map"]`
- [ ] encoder input is exactly 4 channels
- [ ] no zone channels are used in the active pipeline
- [ ] no `nearshore / transition / offshore` in the annotation prompt
- [ ] no zone logic in the active rasterization path
- [ ] no raster prior required for default training
- [ ] no local Qwen required for default annotation
- [ ] annotation uses `OPENAI_API_KEY`
- [ ] annotation JSON contains only `water` and `uncertainty_regions`
- [ ] `semantic_channels.npy` contains exactly one semantic channel: `issue_map`
- [ ] `water_mask` is auxiliary only, not an encoder input channel
- [ ] training can run without `prior_depth_map.npy`
- [ ] logging no longer assumes prior branch outputs always exist

---

## 14. Important consistency rule

Do not leave the codebase in a half-refactored state.

That means the following must all agree with each other:

- `config.yaml`
- `dataset.py`
- `common.py`
- `llm_annotate.py`
- `vlm_utils.py`
- `rasterize_semantics.py`
- `model.py`
- `main.py`

The final codebase must have one clean default mode:

**RGB + single issue_map + OpenAI annotation backend**
