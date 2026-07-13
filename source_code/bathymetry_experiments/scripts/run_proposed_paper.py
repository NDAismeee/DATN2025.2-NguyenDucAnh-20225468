from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    project_root = here.parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=str, default=str(project_root / ".env"))
    parser.add_argument("--config", type=str, default=str(project_root / "configs" / "proposed_paper_agia_napa.yaml"))
    parser.add_argument("--output-dir", type=str, default=str(project_root / "runs_agia_napa"))
    args = parser.parse_args(argv)

    _load_dotenv(Path(args.env_file))

    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from bathymetry_experiments.training.experiment import train_model

    run_dir = train_model("proposed", args.config, args.output_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

