from __future__ import annotations

import _bootstrap  # noqa: F401
import sys
import traceback

import numpy as np
import torch

from bathymetry_llm.models.llm_guided_bathymetry import LLMGuidedBathymetryModel
from bathymetry_llm.models.losses import compute_total_loss
from bathymetry_llm.utils.metrics import (
    compute_metrics,
    interval_coverage,
    mask_iou_f1,
    physical_consistency_metrics,
    uncertainty_diagnostics,
)


def test_model_forward() -> None:
    torch.manual_seed(0)
    b, h, w, k, td, hd = 2, 64, 64, 4, 384, 256
    image = torch.rand(b, 3, h, w)
    unreliable = (torch.rand(b, 1, h, w) > 0.5).float()
    d_phys = torch.rand(b, 1, h, w) * 5.0 + 0.5
    d_min = torch.zeros(b, 1, h, w)
    d_max = d_phys + 1.0
    region_masks = (torch.rand(b, k, h, w) > 0.7).float()
    text_embeddings = torch.randn(b, k, td)
    gamma_map = torch.rand(b, 1, h, w)
    w_phys = (d_max - d_min).clamp_min(0.0)
    region_valid_mask = torch.ones(b, k)
    valid_mask = torch.ones(b, 1, h, w)
    depth = torch.rand(b, 1, h, w) * 5.0

    model = LLMGuidedBathymetryModel(hidden_dim=hd, text_dim=td, gate_hidden=128, gate_dropout=0.1)
    out = model(
        image=image,
        unreliable_mask=unreliable,
        d_phys=d_phys,
        region_masks=region_masks,
        text_embeddings=text_embeddings,
        gamma_map=gamma_map,
        w_phys=w_phys,
        region_valid_mask=region_valid_mask,
    )
    expected = {"depth", "mu", "log_var", "var", "alpha", "z_v", "z_t", "f_vis", "fused", "d_phys"}
    missing = expected - set(out.keys())
    assert not missing, f"missing outputs: {missing}"
    assert out["depth"].shape == (b, 1, h, w)
    assert out["alpha"].min() >= 0 and out["alpha"].max() <= 1
    assert (out["var"] >= 1e-4 - 1e-9).all(), "variance must respect lower bound 1e-4"

    batch = {
        "depth": depth,
        "valid_mask": valid_mask,
        "d_min": d_min,
        "d_max": d_max,
        "gamma_map": gamma_map,
        "region_valid_mask": region_valid_mask,
    }
    losses = compute_total_loss(out, batch, lambda_align=1e-2, lambda_int=1e-2, align_tau=0.07)
    assert "int" in losses and "range" not in losses, "loss key must be 'int', not 'range'"
    losses["total"].backward()
    print("[ok] model_forward + loss + backward")


def test_metrics() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(2.0, 0.5, size=(1, 32, 32)).astype(np.float32)
    pred = target + rng.normal(0.0, 0.1, size=target.shape).astype(np.float32)
    var = (rng.uniform(0.05, 0.5, size=target.shape)).astype(np.float32) ** 2
    valid = (rng.uniform(size=target.shape) > 0.1).astype(np.float32)
    unreliable = (rng.uniform(size=target.shape) > 0.7).astype(np.float32)
    d_min = target - 0.5
    d_max = target + 0.5

    base = compute_metrics(target, pred, valid)
    unc = uncertainty_diagnostics(target, pred, var, valid)
    phys = physical_consistency_metrics(
        target, pred, valid,
        unreliable_mask=unreliable, domain_min=0.0, domain_max=10.0,
    )
    cov = interval_coverage(target, d_min, d_max, valid)
    iou = mask_iou_f1(unreliable, unreliable)
    assert {"mae", "rmse", "err_std"}.issubset(base)
    assert {"ece", "sigma_err_corr", "low_q_mae", "high_q_mae"}.issubset(unc)
    assert {"slope_err", "out_of_range", "unreliable_mae"}.issubset(phys)
    assert cov["coverage"] > 0.5
    assert iou["iou"] == 1.0 and iou["f1"] == 1.0
    print("[ok] metrics: base + uncertainty + physical_consistency + interval_coverage + mask_iou_f1")


def test_validate_llm_output() -> None:
    from bathymetry_llm.llm_pipeline.validate_llm_output import (
        ALLOWED_DISTURBANCE_TYPES,
        ALLOWED_FAILURE_DIRECTIONS,
        validate_llm_output,
    )
    expected_types = {"sun_glint", "shadow", "turbidity", "foam", "ambiguous_bottom", "other"}
    assert ALLOWED_DISTURBANCE_TYPES == expected_types, ALLOWED_DISTURBANCE_TYPES
    assert ALLOWED_FAILURE_DIRECTIONS == {"artificially_shallow", "artificially_deep", "ambiguous"}
    sample = {
        "scene_id": "tile_001",
        "image_assessment": {"overall_condition": "clear", "confidence": 0.7},
        "disturbance_regions": [
            {
                "region_id": "d1",
                "type": "sun_glint",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "severity": 0.6,
                "failure_direction": "artificially_shallow",
                "description": "specular highlights",
                "expected_effect": "appears shallower",
            }
        ],
        "depth_prior_regions": [
            {
                "region_id": "p1",
                "region_name": "nearshore",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "depth_min": 0.0,
                "depth_max": 1.5,
                "rationale": "bright sand near shore",
                "confidence": 0.7,
            }
        ],
        "global_depth_range": {"depth_min": 0.0, "depth_max": 6.0, "unit": "meters"},
        "warnings": [],
    }
    ok, errs = validate_llm_output(sample, height=32, width=32, raise_on_error=False)
    assert ok, errs
    bad = dict(sample)
    bad_regions = [dict(sample["disturbance_regions"][0])]
    bad_regions[0]["failure_direction"] = "wrong"
    bad["disturbance_regions"] = bad_regions
    ok2, errs2 = validate_llm_output(bad, height=32, width=32, raise_on_error=False)
    assert not ok2 and any("failure_direction" in e for e in errs2), errs2
    print("[ok] validate_llm_output: categories + failure_direction enforced")


def test_scheduler() -> None:
    from bathymetry_llm.scripts.train import build_warmup_cosine_scheduler
    model = torch.nn.Linear(2, 2)
    base_lr = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    sched = build_warmup_cosine_scheduler(opt, total_epochs=10, warmup_epochs=3, base_lr=base_lr)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] < lrs[2], lrs
    assert lrs[-1] < lrs[3], lrs
    print(f"[ok] scheduler: warmup lrs={[f'{x:.2e}' for x in lrs[:4]]} tail={[f'{x:.2e}' for x in lrs[-3:]]}")


def main() -> int:
    failed = 0
    for fn in (test_model_forward, test_metrics, test_validate_llm_output, test_scheduler):
        try:
            fn()
        except Exception:
            failed += 1
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
    if failed:
        print(f"\n{failed} test(s) FAILED")
        return 1
    print("\nAll smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
