from bathymetry_experiments.cli import main

for model in ["proposed", "cnn", "knn", "depth_anything_v2", "unet"]:
    main(["train", "--model", model, "--dry-run"])
