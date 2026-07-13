from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


ROOT = Path("/mnt/disk3/anhnd2468/MagicBathyNet/hao-chapter1-depth-prediction/source_code")

IMAGE_DIR = "/mnt/disk3/anhnd2468/MagicBathyNet/agia_napa/img/aerial"
DEPTH_DIR = "/mnt/disk3/anhnd2468/MagicBathyNet/agia_napa/depth/aerial"

CONDA_ENV = "ndapy312"


@dataclass
class Step:
    cwd: Path
    argv: List[str]
    gpu: bool


@dataclass
class ModelRun:
    name: str
    steps: List[Step]


def _run_step(step: Step) -> Tuple[int, str]:
    env = os.environ.copy()
    env["IMAGE_DIR"] = IMAGE_DIR
    env["DEPTH_DIR"] = DEPTH_DIR
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = ["conda", "run", "-n", CONDA_ENV] + step.argv
    p = subprocess.run(
        cmd,
        cwd=str(step.cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout


def _is_oom(output: str) -> bool:
    s = output.lower()
    return (
        "cuda out of memory" in s
        or "cublas_status_alloc_failed" in s
        or "outofmemoryerror" in s
        or "resource exhausted" in s
    )


def _print_cmd(step: Step) -> None:
    cmd = ["conda", "run", "-n", CONDA_ENV] + step.argv
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"$ (cd {step.cwd}) {pretty}")


def build_runs() -> List[ModelRun]:
    runs: List[ModelRun] = []

    # CNN (already done earlier, but included for completeness)
    runs.append(
        ModelRun(
            name="cnn_src",
            steps=[
                Step(
                    cwd=ROOT / "cnn_src",
                    argv=["python", "train.py", "--config", "config_single369_10ep.yaml", "--patch_stem", "img_369", "--device", "cuda"],
                    gpu=True,
                ),
                Step(
                    cwd=ROOT / "cnn_src",
                    argv=[
                        "python",
                        "infer.py",
                        "--config",
                        "config_single369_10ep.yaml",
                        "--checkpoint",
                        "checkpoints_cnn_rgb_single369_10ep/best_model.pt",
                        "--patch_stem",
                        "img_379",
                        "--output_dir",
                        "cnn_infer_outputs_after_single369_10ep_turboPalette",
                        "--device",
                        "cuda",
                    ],
                    gpu=True,
                ),
            ],
        )
    )

    runs.append(
        ModelRun(
            name="da-sdb",
            steps=[
                Step(
                    cwd=ROOT / "da-sdb",
                    argv=["python", "train.py", "--config", "config_single369_10ep.yaml", "--device", "cuda"],
                    gpu=True,
                ),
                Step(
                    cwd=ROOT / "da-sdb",
                    argv=["python", "infer.py", "--config", "config_infer379.yaml", "--device", "cuda"],
                    gpu=True,
                ),
            ],
        )
    )

    runs.append(
        ModelRun(
            name="depth_anythingv2",
            steps=[
                Step(
                    cwd=ROOT / "depth_anythingv2",
                    argv=["python", "infer.py", "--config", "config_infer379.yaml"],
                    gpu=True,
                )
            ],
        )
    )

    runs.append(
        ModelRun(
            name="dpt",
            steps=[
                Step(
                    cwd=ROOT / "dpt",
                    argv=["python", "train.py", "--config", "config_single369_10ep.yaml", "--device", "cuda"],
                    gpu=True,
                ),
                Step(
                    cwd=ROOT / "dpt",
                    argv=["python", "infer.py", "--config", "config_infer379.yaml", "--device", "cuda"],
                    gpu=True,
                ),
            ],
        )
    )

    runs.append(
        ModelRun(
            name="mlp",
            steps=[
                Step(
                    cwd=ROOT / "mlp",
                    argv=["python", "train.py", "--config", "config_single369_10iter.yaml"],
                    gpu=False,
                ),
                Step(
                    cwd=ROOT / "mlp",
                    argv=["python", "infer.py", "--config", "config_infer379.yaml"],
                    gpu=False,
                ),
            ],
        )
    )

    runs.append(
        ModelRun(
            name="rf",
            steps=[
                Step(
                    cwd=ROOT / "rf",
                    argv=["python", "train.py", "--config", "config_single369.yaml"],
                    gpu=False,
                ),
                Step(
                    cwd=ROOT / "rf",
                    argv=["python", "infer.py", "--config", "config_infer379.yaml"],
                    gpu=False,
                ),
            ],
        )
    )

    runs.append(
        ModelRun(
            name="unet",
            steps=[
                Step(
                    cwd=ROOT / "unet",
                    argv=["python", "train.py", "--config", "config_single369_10ep.yaml", "--device", "cuda"],
                    gpu=True,
                ),
                Step(
                    cwd=ROOT / "unet",
                    argv=["python", "infer.py", "--config", "config_infer379.yaml", "--device", "cuda"],
                    gpu=True,
                ),
            ],
        )
    )

    return runs


def main() -> None:
    print("IMAGE_DIR:", IMAGE_DIR)
    print("DEPTH_DIR:", DEPTH_DIR)
    print("CONDA_ENV:", CONDA_ENV)
    print()

    successes: List[str] = []
    skipped: List[str] = []
    failed: List[str] = []

    for run in build_runs():
        print("=" * 80)
        print("MODEL:", run.name)
        ok = True
        for step in run.steps:
            _print_cmd(step)
            code, out = _run_step(step)
            print(out)
            if code != 0:
                if step.gpu and _is_oom(out):
                    msg = f"{run.name}: OOM (skipped remaining steps)"
                    print(msg)
                    skipped.append(msg)
                else:
                    msg = f"{run.name}: failed (exit_code={code})"
                    print(msg)
                    failed.append(msg)
                ok = False
                break
        if ok:
            successes.append(run.name)

    print("\n" + "=" * 80)
    print("DONE")
    print("Success:", successes)
    print("Skipped:", skipped)
    print("Failed:", failed)


if __name__ == "__main__":
    main()

