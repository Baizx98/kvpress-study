from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "evaluation" / "ATC26_postprocess_prefill_sweep.py"
EXPERIMENT_NAME = "ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv"


def load_base_module():
    spec = importlib.util.spec_from_file_location("atc26_base_postprocess", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_base_module()
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
    base.ARTIFACTS_DIR = base.RESULT_ROOT / "artifacts"
    base.RAW_DIR = base.ARTIFACTS_DIR / "raw"
    base.METRICS_LONG = base.ARTIFACTS_DIR / "ATC26_metrics_long.csv"
    base.METRICS_WIDE = base.ARTIFACTS_DIR / "ATC26_metrics_wide.csv"
    base.METRICS_FULL_LONG = base.ARTIFACTS_DIR / "ATC26_metrics_full_long.csv"
    base.METRICS_FULL_WIDE = base.ARTIFACTS_DIR / "ATC26_metrics_full_wide.csv"
    base.JOB_STATUS = base.ARTIFACTS_DIR / "ATC26_job_status.csv"
    base.RESULT_README = base.RESULT_ROOT / "README.md"
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
