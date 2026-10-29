"""Persistencia de resultados de backtest y bitácora de iteraciones."""
import json
import platform

import numpy as np
import pandas as pd

from . import config as C

RUNS = C.RESULTS_DIR / "runs"
ITER_CSV = C.RESULTS_DIR / "iterations.csv"
EPOCH_CSV = C.RESULTS_DIR / "epoch_logs.csv"


def hardware():
    import torch
    dev = "MPS" if torch.backends.mps.is_available() else "CPU"
    return f"Apple M4 Pro ({dev}), {platform.machine()}"


def save_run(res, meta, split="test"):
    """Guarda F/A/Q de un backtest + metadatos (params, tiempos, L, estrategia...)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    name = f"{split}__{res['name']}"
    arrs = dict(F=res["F"].astype("float32"), A=res["A"].astype("float32"),
                origins=res["origins"])
    if "Q" in res:
        arrs["Q"] = res["Q"].astype("float32")
        meta["quantiles"] = list(res["quantiles"])
    np.savez_compressed(RUNS / f"{name}.npz", **arrs)
    meta = dict(meta, name=res["name"], split=split, infer_time=res["infer_time"])
    meta.setdefault("hardware", hardware())
    (RUNS / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))


def load_run(name, split="test"):
    z = np.load(RUNS / f"{split}__{name}.npz")
    res = {k: z[k] for k in z.files}
    meta = json.loads((RUNS / f"{split}__{name}.json").read_text())
    res.update(name=name, infer_time=meta["infer_time"], meta=meta)
    if "quantiles" in meta:
        res["quantiles"] = meta["quantiles"]
    return res


def list_runs(split="test"):
    return sorted(p.stem.split("__", 1)[1] for p in RUNS.glob(f"{split}__*.json"))


def log_iteration(info, epochs=None, group=""):
    """Agrega una fila a iterations.csv y las curvas por epoch a epoch_logs.csv."""
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in info.items() if not isinstance(v, (dict, list))}
    row["group"] = group
    new = pd.DataFrame([row])
    if ITER_CSV.exists():                    # unión de columnas (modelos con distintos campos)
        new = pd.concat([pd.read_csv(ITER_CSV), new], ignore_index=True)
    new.to_csv(ITER_CSV, index=False)
    if epochs:
        e = pd.DataFrame(epochs)
        e.insert(0, "tag", info["tag"])
        e.insert(1, "group", group)
        e.to_csv(EPOCH_CSV, mode="a", header=not EPOCH_CSV.exists(), index=False)
