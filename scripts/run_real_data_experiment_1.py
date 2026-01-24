import hashlib
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _enforce_tf_determinism() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        import tensorflow as tf  # noqa: F401

        enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
        if callable(enable_determinism):
            enable_determinism()
    except Exception:
        pass


def _sha256_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()

    for c in df.columns:
        h.update(str(c).encode("utf-8"))
        h.update(b"\0")

    if isinstance(df.index, pd.DatetimeIndex):
        h.update(df.index.view("int64").tobytes())
    else:
        h.update(pd.Index(df.index).astype(str).to_numpy().tobytes())

    arr = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
    h.update(arr.tobytes())

    return h.hexdigest()


def main() -> int:
    _enforce_tf_determinism()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.real_data_experiment_config import run_real_data_experiment_from_config

    config_path = Path("configs/real_data_experiment_1.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    out1 = run_real_data_experiment_from_config(str(config_path))
    out2 = run_real_data_experiment_from_config(str(config_path))

    run_id_1 = out1.get("run_id")
    run_id_2 = out2.get("run_id")

    if not isinstance(run_id_1, str) or not re.fullmatch(r"[0-9a-f]{64}", run_id_1):
        raise ValueError(f"run_id is not a 64-char hex string: {run_id_1!r}")

    result_1 = out1["result"]
    health_1 = result_1["health_report"]
    train_eval_1 = result_1["train_eval"]

    print("A) run_id")
    print(run_id_1)
    print()

    print("B) health_gates")
    print(f"passed={health_1.get('passed')}")
    print(f"failed_gates={health_1.get('failed_gates')}")
    print()

    print("C) dataset_row_counts")
    print(f"n_rows_xy_before_drop={result_1.get('n_rows_xy_before_drop')}")
    print(f"n_rows_xy_after_drop={result_1.get('n_rows_xy_after_drop')}")
    print()

    print("D) split_sizes")
    print(f"n_train={train_eval_1.get('n_train')}")
    print(f"n_val={train_eval_1.get('n_val')}")
    print(f"n_test={train_eval_1.get('n_test')}")
    print()

    print("E) baseline_metrics")
    print(train_eval_1.get("metrics"))
    print()

    y_pred_1 = train_eval_1["y_pred_test"]
    y_pred_2 = out2["result"]["train_eval"]["y_pred_test"]

    hash_1 = _sha256_df(y_pred_1)
    hash_2 = _sha256_df(y_pred_2)

    print("F) determinism_check")
    print(f"run_id_same={run_id_1 == run_id_2}")
    print(f"y_pred_test_hash_1={hash_1}")
    print(f"y_pred_test_hash_2={hash_2}")
    print(f"y_pred_test_identical={(hash_1 == hash_2) and y_pred_1.equals(y_pred_2)}")

    if run_id_1 != run_id_2:
        raise ValueError(f"Determinism failure: run_id differs ({run_id_1} != {run_id_2})")
    if (hash_1 != hash_2) or (not y_pred_1.equals(y_pred_2)):
        raise ValueError("Determinism failure: y_pred_test differs across identical runs")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
