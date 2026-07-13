from __future__ import annotations

import torch

from common import load_yaml_config
from model import LLMGuidedBathymetryModel
from rasterize_semantics import rasterize_annotation
from vlm_utils import validate_annotation_schema


def test_model_forward() -> None:
    config = load_yaml_config("config.yaml")
    config.setdefault("model", {})
    config.setdefault("debug", {})
    config.setdefault("train", {})
    config["model"]["latent_dim"] = 32
    config["model"]["align_dim"] = 16
    config["model"]["gating_hidden_dim"] = 16
    config["model"]["decoder_feature_dim"] = 16
    config["model"]["text_dim"] = 384
    config["model"]["use_reconstruction_loss"] = True
    config["train"]["lambda_recon"] = 0.05
    config["debug"]["assert_local_prior"] = True
    model = LLMGuidedBathymetryModel(config)
    model.eval()

    batch_size, height, width, k_regions, text_dim = 2, 16, 16, 3, 384
    image = torch.rand(batch_size, 3, height, width)
    reliability = torch.zeros(batch_size, 1, height, width)
    reliability[:, :, :8, :8] = 1.0
    disturbance = torch.zeros(batch_size, k_regions, height, width)
    disturbance[:, 0, :8, :8] = 1.0
    disturbance[:, 1, 8:, :8] = 1.0
    disturbance[:, 2, :, 8:] = 1.0
    text = torch.rand(batch_size, k_regions, text_dim)
    region_valid = torch.zeros(batch_size, k_regions)
    region_valid[:, 0] = 1.0
    prior = torch.zeros(batch_size, 1, height, width)
    prior[:, :, :8, :8] = 2.0
    prior_valid = torch.zeros(batch_size, 1, height, width)
    prior_valid[:, :, :8, :8] = 1.0
    prior_conf = torch.full((batch_size, 1, height, width), 0.7) * prior_valid
    depth = torch.full((batch_size, 1, height, width), 3.0)
    mask = torch.ones(batch_size, 1, height, width)

    pred, info = model(
        image,
        reliability_mask=reliability,
        disturbance_masks=disturbance,
        text_embeddings=text,
        region_valid_mask=region_valid,
        prior_depth_map=prior,
        prior_valid_mask=prior_valid,
        prior_confidence=prior_conf,
        water_mask=mask,
        depth_gt=depth,
        valid_mask=mask,
    )

    # Check output shapes
    assert pred.shape == (batch_size, 1, height, width)
    assert info["mu"].shape == pred.shape
    assert info["var"].min().item() >= 1.0e-4
    assert info["alpha_eff"].shape == pred.shape
    
    # Check bottleneck
    assert "latent" in info
    latent = info["latent"]
    assert latent.shape[0] == batch_size
    assert latent.shape[-2] < height
    assert latent.shape[-1] < width
    expected_latent_h = (height + 3) // 4
    expected_latent_w = (width + 3) // 4
    assert latent.shape[-2] == expected_latent_h
    assert latent.shape[-1] == expected_latent_w
    
    # Check reconstruction output
    assert "reconstruction" in info
    reconstruction = info["reconstruction"]
    assert reconstruction.shape == (batch_size, 3, height, width)
    assert torch.isfinite(reconstruction).all()
    
    # Check decoded features
    assert "decoded_features" in info
    decoded_features = info["decoded_features"]
    assert decoded_features.shape == (batch_size, 16, height, width)
    
    # Check reconstruction loss
    assert "recon_loss" in info
    assert torch.isfinite(info["recon_loss"])
    assert info["recon_loss"].item() >= 0.0
    
    assert torch.max(info["alpha_eff"][:, :, 8:, 8:]) < 1e-5
    assert torch.all(info["d_phys"][:, :, 8:, 8:] == 0)
    assert torch.isfinite(info["align_loss"])
    
    # Check total loss includes reconstruction
    expected = (
        info["nll_loss"] 
        + float(config["train"]["lambda_align"]) * info["align_loss"]
        + float(config["train"]["lambda_recon"]) * info["recon_loss"]
    )
    assert torch.allclose(info["total"], expected, atol=1.0e-6)

    prior_conf_low = prior_conf.clone()
    prior_conf_low[:, :, :8, :8] = 0.1
    _, info_low = model(
        image,
        reliability_mask=reliability,
        disturbance_masks=disturbance,
        text_embeddings=text,
        region_valid_mask=region_valid,
        prior_depth_map=prior,
        prior_valid_mask=prior_valid,
        prior_confidence=prior_conf_low,
        water_mask=mask,
    )
    assert info_low["alpha_eff"].mean() < info["alpha_eff"].mean()


def test_schema_strict() -> None:
    ok, _ = validate_annotation_schema(
        {
            "water": {"polygons": [[[0, 0], [10, 0], [10, 10]]]},
            "problem_regions": [],
            "scene_summary": "clean water",
        }
    )
    assert ok

    bad_legacy, msg = validate_annotation_schema(
        {
            "water": {"polygons": [[[0, 0], [10, 0], [10, 10]]]},
            "uncertainty_regions": [{"issue_type": "shadow", "polygons": [[[1, 1], [2, 1], [2, 2]]]}],
        }
    )
    assert not bad_legacy
    assert "legacy field" in msg

    bad_depth, msg = validate_annotation_schema(
        {
            "water": {"polygons": [[[0, 0], [10, 0], [10, 10]]]},
            "problem_regions": [
                {
                    "category": "shadow",
                    "polygons": [[[1, 1], [5, 1], [5, 5]]],
                    "severity": 0.8,
                    "description": "shadow region",
                    "rationale": "nearshore",
                }
            ],
            "scene_summary": "bad",
        }
    )
    assert not bad_depth
    assert "depth_min" in msg


def test_rasterize_empty_and_local_only() -> None:
    anno = {
        "water": {"polygons": [[[0, 0], [9, 0], [9, 9], [0, 9]]]},
        "problem_regions": [],
        "scene_summary": "clean water",
    }
    out = rasterize_annotation(anno, 10, 10, 0.0, 30.29)
    assert out["R"].shape == (1, 10, 10)
    assert out["M"].sum() == 0
    assert out["prior_valid"].sum() == 0
    assert out["prior"].sum() == 0

    config = load_yaml_config("config.yaml")
    config.setdefault("model", {})
    config.setdefault("train", {})
    config["model"]["use_reconstruction_loss"] = True
    config["train"]["lambda_recon"] = 0.05
    model = LLMGuidedBathymetryModel(config)
    model.eval()
    image = torch.rand(1, 3, 10, 10)
    reliability = torch.zeros(1, 1, 10, 10)
    water = torch.ones(1, 1, 10, 10)
    pred, info = model(
        image,
        reliability_mask=reliability,
        disturbance_masks=torch.zeros(1, 1, 10, 10),
        text_embeddings=torch.zeros(1, 1, 384),
        region_valid_mask=torch.zeros(1, 1),
        prior_depth_map=torch.zeros(1, 1, 10, 10),
        prior_valid_mask=torch.zeros(1, 1, 10, 10),
        prior_confidence=torch.zeros(1, 1, 10, 10),
        water_mask=water,
    )
    assert torch.allclose(pred, info["mu"] * water)
    
    # Check reconstruction is generated even with no regions
    assert "reconstruction" in info
    assert info["reconstruction"].shape == (1, 3, 10, 10)


def main() -> None:
    test_schema_strict()
    test_rasterize_empty_and_local_only()
    test_model_forward()
    print("sanity_check passed")


if __name__ == "__main__":
    main()
