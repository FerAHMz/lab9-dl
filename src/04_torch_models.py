"""Etapa 4: modelos propios en PyTorch (MLP, DLinear, NLinear, LSTM x3 estrategias, seq2seq).

1) Iteraciones de hiperparámetros evaluadas SOLO en validación (ventana L, meses de
   historia, normalización, tamaño oculto, capas, dropout, lr, batch).
2) Para cada modelo se elige la mejor configuración por MASE de validación y se evalúa
   UNA vez en prueba con backtest() (92 orígenes).
Se puede relanzar: las iteraciones ya registradas se saltan (checkpoints en results/ckpt).
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd                                                    # noqa: E402
import torch                                                           # noqa: E402

from src import config as C                                            # noqa: E402
from src.evaluation import backtest                                    # noqa: E402
from src.torch_models import MODELS                                    # noqa: E402
from src.train_torch import (Normalizer, Panel, TorchForecaster,       # noqa: E402
                             TrainConfig, get_device, train_model)
from src.utils import ITER_CSV, log_iteration, save_run                # noqa: E402
from src.train_torch import _model_kwargs                              # noqa: E402

CKPT = C.RESULTS_DIR / "ckpt"
BASE = dict(epochs=25, samples_per_epoch=32768, patience=4, batch=512)


def grid():
    """Lista (grupo, config) de todas las iteraciones."""
    g = []
    lstm = dict(model="LSTM-Seq2Seq", hidden=128, layers=2, **{**BASE, "epochs": 15,
                                                                 "patience": 3})
    # -- ventana L
    for m in ("DLinear", "LSTM-Seq2Seq"):
        for L in (48, 168, 336):
            g.append(("window_L", TrainConfig(**{**lstm, "model": m, "L": L})))
    # -- meses de historia
    for m in ("DLinear", "LSTM-Seq2Seq"):
        for months in (3, 12, 30):
            g.append(("history", TrainConfig(**{**lstm, "model": m, "months": months})))
    # -- normalización
    for m in ("DLinear", "LSTM-Seq2Seq"):
        for norm in ("series", "global", "window"):
            g.append(("norm", TrainConfig(**{**lstm, "model": m, "norm": norm})))
    # -- hiperparámetros de la LSTM
    for kw in (dict(hidden=64), dict(hidden=256), dict(layers=1), dict(dropout=0.3),
               dict(lr=3e-3)):
        g.append(("lstm_hp", TrainConfig(**{**lstm, **kw})))
    # -- estrategias multi-paso con la misma LSTM
    for m in ("LSTM-AR", "LSTM-MIMO", "LSTM-Seq2Seq"):
        g.append(("strategy", TrainConfig(**{**lstm, "model": m})))
    g.append(("strategy", TrainConfig(**{**lstm, "extra": {"feed_prev": True}},)))
    # -- MLP y lineales
    for kw in (dict(lr=1e-3), dict(lr=3e-4), dict(dropout=0.3), dict(L=336)):
        g.append(("mlp", TrainConfig(**{**BASE, "model": "MLP", **kw})))
    for L in (168, 336):
        g.append(("linear", TrainConfig(**{**BASE, "model": "NLinear", "L": L})))
    # -- H = 168 (lineales y MLP), para el MASE semanal de la tabla final
    for m in ("DLinear", "NLinear", "MLP"):
        g.append(("H168", TrainConfig(**{**BASE, "model": m, "L": 336, "H": 168})))
    # -- presupuesto largo e igual para todos (las corridas anteriores tocaban el tope)
    long = dict(BASE, epochs=40, patience=5, samples_per_epoch=32768)
    for m, ex in (("LSTM-AR", {}), ("LSTM-MIMO", {}), ("LSTM-Seq2Seq", {}),
                  ("LSTM-Seq2Seq", {"feed_prev": True})):
        g.append(("long", TrainConfig(**{**long, "model": m, "extra": {**ex, "long": True}})))
    g.append(("long", TrainConfig(**{**long, "model": "MLP", "L": 336, "epochs": 60,
                                     "extra": {"long": True}})))
    return g


def done_tags():
    if not ITER_CSV.exists():
        return set()
    return set(pd.read_csv(ITER_CSV)["tag"])


def tag_of(cfg):
    return cfg.tag()


def run_one(k, device):
    """Entrena la iteración k del grid (se llama en un subproceso aparte)."""
    P = Panel()
    group, cfg = grid()[k]
    print(f"[{group}] {cfg.tag()} ({device})", flush=True)
    fc, log = train_model(cfg, P, device)
    torch.save(fc.net.state_dict(), CKPT / f"{cfg.tag()}.pt")
    log_iteration(log["info"], log["epochs"], group=group)


def run_all():
    """Cada iteración corre en su propio proceso: la memoria del backend MPS (que crece
    con la LSTM) se libera al terminar, y el tope de memoria evita saturar la RAM. Si una
    iteración se queda sin memoria en MPS, se repite en CPU."""
    import subprocess
    CKPT.mkdir(parents=True, exist_ok=True)
    for k, (group, cfg) in enumerate(grid()):
        if cfg.tag() in done_tags():
            continue
        cmd = [sys.executable, "-u", __file__, "--one", str(k)]
        if subprocess.run(cmd).returncode != 0:
            print(f"  reintento en CPU: {cfg.tag()}", flush=True)
            subprocess.run(cmd + ["--cpu"])


def load_forecaster(row, P, device, name):
    """Reconstruye el forecaster de una iteración registrada (pesos + normalizador)."""
    import json
    extra = json.loads(row["extra"]) if isinstance(row["extra"], str) else {}
    cfg = TrainConfig(model=row["model"], L=int(row["L"]), H=int(row["H"]),
                      months=int(row["months"]), norm=row["norm"], hidden=int(row["hidden"]),
                      layers=int(row["layers"]), dropout=float(row["dropout"]),
                      lr=float(row["lr"]), batch=int(row["batch"]), seed=int(row["seed"]),
                      extra=extra)
    net = MODELS[cfg.model](cfg.L, cfg.H, **_model_kwargs(cfg)).to(device)
    net.load_state_dict(torch.load(CKPT / f"{row['tag']}.pt", map_location=device))
    t_start = max(0, P.t_train_end + 1 - cfg.months * 30 * 24 - cfg.L)
    norm = Normalizer(cfg.norm, P.Y[t_start:P.t_train_end + 1])
    return TorchForecaster(name, net, cfg, P, norm, device), cfg


FINAL = {  # nombre en la tabla -> (modelo, familia)
    "MLP": ("MLP", "MLP y lineales"), "DLinear": ("DLinear", "MLP y lineales"),
    "NLinear": ("NLinear", "MLP y lineales"), "LSTM-MIMO": ("LSTM-MIMO", "Recurrentes"),
    "LSTM-AR": ("LSTM-AR", "Recurrentes"), "LSTM-Seq2Seq": ("LSTM-Seq2Seq", "Encoder-decoder"),
}


def evaluate_best(P, device):
    it = pd.read_csv(ITER_CSV)
    it = it[it["H"] == C.H]
    best_rows = {}
    for name, (model, family) in FINAL.items():
        rows = it[it["model"] == model].sort_values("best_val_MASE")
        row = rows.iloc[0]
        best_rows[name] = row
        fc, cfg = load_forecaster(row, P, device, name)
        res = backtest(fc, P.Y, P.test_origins, scale=P.scale)
        save_run(res, dict(family=family, L=cfg.L, strategy=MODELS[model].strategy,
                           covariates="Sí" if MODELS[model].uses_cov else "No",
                           params_total=int(row["params_total"]),
                           params_trainable=int(row["params_trainable"]),
                           train_time=float(row["train_time"]), months=cfg.months,
                           tag=row["tag"], val_MASE=float(row["best_val_MASE"]),
                           peak_mem_MB=float(row["peak_mem_MB"])))
        print(f"{name:14s} {row['tag']}: val {row['best_val_MASE']:.4f} -> "
              f"test MASE {res['metrics']['MASE']:.4f}", flush=True)
    # H = 168: mismo backtest diario, con orígenes cuyo horizonte semanal cabe en prueba
    o168 = P.test_origins[P.test_origins + 168 <= P.t_test_end + 1]
    it168 = pd.read_csv(ITER_CSV)
    for _, row in it168[it168["H"] == 168].iterrows():
        fc, cfg = load_forecaster(row, P, device, f"{row['model']}_H168")
        res = backtest(fc, P.Y, o168, H=168, scale=P.scale)
        save_run(res, dict(tag=row["tag"], H=168), split="test168")
        print(f"{row['model']} H=168 test MASE {res['metrics']['MASE']:.4f}")


if __name__ == "__main__":
    if "--one" in sys.argv:
        dev = torch.device("cpu") if "--cpu" in sys.argv else get_device()
        if dev.type == "cpu":
            torch.set_num_threads(8)
        run_one(int(sys.argv[sys.argv.index("--one") + 1]), dev)
    elif "--eval" in sys.argv:
        evaluate_best(Panel(), get_device())
    else:
        run_all()
