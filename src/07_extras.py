"""Etapa 7: experimentos complementarios sobre el mejor modelo propio.

  --retrain-nf / --retrain-lgbm / --retrain-mstl  reentrenamiento al inicio de cada mes de prueba (oct, nov, dic) con ventana
              expandible, para el mejor modelo DL propio y para LightGBM / MSTL.
  --local     modelo local (uno por cliente, 10 clientes) vs. global.
  --seeds     mejor modelo con 3 semillas -> desviación estándar del MASE.
  --h168      MASE con H = 168 para los seasonal naive (numpy).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "--retrain-lgbm" in sys.argv:   # LightGBM antes que torch (choque de OpenMP en macOS)
    import lightgbm  # noqa: E402,F401
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from src import config as C                                           # noqa: E402
from src.evaluation import Forecaster, backtest                       # noqa: E402
from src.train_torch import Panel, TrainConfig, get_device, train_model  # noqa: E402
from src.utils import ITER_CSV, save_run                              # noqa: E402

MONTHS = ("2014-10-01", "2014-11-01", "2014-12-01")
VAL_DAYS = 30


def best_cfg(model=None):
    """Mejor configuración propia (H=24) según MASE de validación."""
    it = pd.read_csv(ITER_CSV)
    it = it[(it["H"] == C.H) & it["model"].isin(
        ["MLP", "DLinear", "NLinear", "LSTM-MIMO", "LSTM-AR", "LSTM-Seq2Seq"])]
    if model:
        it = it[it["model"] == model]
    r = it.sort_values("best_val_MASE").iloc[0]
    import json
    extra = json.loads(r["extra"]) if isinstance(r["extra"], str) else {}
    return TrainConfig(model=r["model"], L=int(r["L"]), H=int(r["H"]), months=int(r["months"]),
                       norm=r["norm"], hidden=int(r["hidden"]), layers=int(r["layers"]),
                       dropout=float(r["dropout"]), lr=float(r["lr"]), batch=int(r["batch"]),
                       epochs=int(r["epochs"]), samples_per_epoch=int(r["samples_per_epoch"]),
                       patience=int(r["patience"]), extra=extra)


def month_blocks(P):
    """(t_inicio_mes, orígenes del mes) para oct, nov y dic."""
    out = []
    for k, m in enumerate(MONTHS):
        t_ms = int(P.index.get_loc(pd.Timestamp(m)))
        t_me = int(P.index.get_loc(pd.Timestamp(MONTHS[k + 1]))) if k < 2 else P.t_test_end + 1
        out.append((t_ms, P.test_origins[(P.test_origins >= t_ms) & (P.test_origins < t_me)]))
    return out


class Stitched(Forecaster):
    """Concatena forecasters entrenados en distintos cortes: usa el del mes del origen."""

    def __init__(self, name, parts):
        self.name, self.parts = name, parts            # [(origins, forecaster)]

    def forecast(self, hist, t0):
        for origins, fc in self.parts:
            if t0 in origins:
                return fc.forecast(hist, t0)
        raise KeyError(t0)


def nf_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s05", Path(__file__).resolve().parent / "05_neuralforecast.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def retrain_nf(P):
    """Mejor DL (iTransformer) y mejores baselines reentrenados al inicio de cada mes con
    ventana expandible; los últimos 92 días antes del mes son el bloque de early stopping."""
    from src.library_models import NFForecaster, nf_fit, train_stats
    s05 = nf_module()
    parts, t_tot = [], 0.0
    for t_ms, origins in month_blocks(P):
        stats = train_stats(P, t_ms - 1 - s05.VAL)          # estadísticos sin el bloque de ES
        nf, tt = nf_fit([s05.build("iTransformer")], P, t_ms - 1, val_size=s05.VAL, stats=stats)
        t_tot += tt
        parts.append((set(int(o) for o in origins), NFForecaster(nf, P, "iTransformer", *stats)))
        print(f"  iTransformer reentrenado al {P.index[t_ms].date()} ({tt:.0f}s)", flush=True)
    res = backtest(Stitched("iTransformer_retrain", parts), P.Y, P.test_origins, scale=P.scale)
    save_run(res, dict(train_time=t_tot, base="iTransformer"), split="retrain")
    print(f"iTransformer reentrenado mensual: MASE {res['metrics']['MASE']:.4f}", flush=True)


def retrain_lgbm(P):
    from src.library_models import LGBMForecaster
    parts, t_tot = [], 0.0
    for t_ms, origins in month_blocks(P):
        m = LGBMForecaster(P, months=999, name="LightGBM")
        m.fit(t_train_end=t_ms - 1 - VAL_DAYS * 24, t_val=(t_ms - VAL_DAYS * 24, t_ms - 1))
        t_tot += m.train_time
        parts.append((set(int(o) for o in origins), m))
    res = backtest(Stitched("LightGBM_retrain", parts), P.Y, P.test_origins, scale=P.scale)
    save_run(res, dict(train_time=t_tot, base="LightGBM"), split="retrain")
    print(f"LightGBM reentrenado mensual: MASE {res['metrics']['MASE']:.4f}", flush=True)


def retrain_mstl(P):
    # MSTL (mejor baseline estadístico): se reajusta al inicio de cada mes y se congela
    # (refit=False dentro del mes) en lugar de una sola vez el 30-sep.
    from src.library_models import stats_backtest
    from statsforecast.models import MSTL, AutoETS
    parts = []
    for t_ms, origins in month_blocks(P):
        fcs, _ = stats_backtest(P, origins, models=[MSTL(season_length=[24, 168],
                                trend_forecaster=AutoETS(model="ZZN"), alias="MSTL")])
        parts.append((set(int(o) for o in origins), fcs["MSTL"]))
    res = backtest(Stitched("MSTL_retrain", parts), P.Y, P.test_origins, scale=P.scale)
    save_run(res, dict(base="MSTL"), split="retrain")
    print(f"MSTL reentrenado mensual: MASE {res['metrics']['MASE']:.4f}", flush=True)


def local_vs_global(P, device):
    cfg = best_cfg()
    rng = np.random.default_rng(C.SEED)
    clients = sorted(rng.choice(P.N, 10, replace=False))
    rows = []
    for i in clients:
        c = TrainConfig(**{**cfg.__dict__, "series": [int(i)],
                           "samples_per_epoch": max(4096, cfg.samples_per_epoch // 10)})
        fc, log = train_model(c, P, device, verbose=False)
        res = backtest(fc, P.Y, P.test_origins)
        mae = np.abs(res["F"][0] - res["A"][i]).mean()   # A trae todas las series; fila i
        rows.append(dict(client=P.columns[i], idx=int(i), local_MASE=mae / P.scale[i],
                         local_train_time=log["info"]["train_time"],
                         mean_kwh=float(P.Y[:P.t_train_end + 1, i].mean())))
        print(f"  local {P.columns[i]}: MASE {rows[-1]['local_MASE']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "local_vs_global.csv", index=False)


def seeds_nf(P):
    """Mejor modelo entrenado (iTransformer): semillas 42 (ya evaluada), 43 y 44."""
    from src.utils import load_run
    from src.evaluation import metrics
    s05 = nf_module()
    r = load_run("iTransformer")
    rows = [dict(model="iTransformer", seed=42, test_MASE=metrics(r["F"], r["A"], P.scale)["MASE"],
                 test_MAE=metrics(r["F"], r["A"], P.scale)["MAE"])]
    for s in (43, 44):
        res, _ = s05.fit_eval("iTransformer", P, P.t_test - 1, P.test_origins, "seed", seed=s,
                              alias=f"iTransformer_s{s}")
        rows.append(dict(model="iTransformer", seed=s, test_MASE=res["metrics"]["MASE"],
                         test_MAE=res["metrics"]["MAE"]))
        print(f"  iTransformer semilla {s}: MASE {rows[-1]['test_MASE']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "seeds_itransformer.csv", index=False)


def seeds(P, device):
    cfg = best_cfg()
    rows = []
    for s in (42, 43, 44):
        c = TrainConfig(**{**cfg.__dict__, "seed": s})
        fc, log = train_model(c, P, device, verbose=False)
        res = backtest(fc, P.Y, P.test_origins, scale=P.scale)
        rows.append(dict(model=cfg.model, seed=s, val_MASE=log["info"]["best_val_MASE"],
                         test_MASE=res["metrics"]["MASE"], test_MAE=res["metrics"]["MAE"]))
        print(f"  semilla {s}: test MASE {rows[-1]['test_MASE']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "seeds.csv", index=False)


class SNaiveH(Forecaster):
    def __init__(self, m, H):
        self.m, self.H, self.name = m, H, f"SeasonalNaive{m}_H{H}"

    def forecast(self, hist, t0):
        reps = int(np.ceil(self.H / self.m))
        return np.tile(hist[-self.m:], (reps, 1))[:self.H].T


def h168(P):
    o168 = P.test_origins[P.test_origins + 168 <= P.t_test_end + 1]
    for m in (24, 168):
        res = backtest(SNaiveH(m, 168), P.Y, o168, H=168, scale=P.scale)
        save_run(res, dict(H=168), split="test168")
        print(f"SNaive{m} H=168 MASE {res['metrics']['MASE']:.4f}")


if __name__ == "__main__":
    P, device = Panel(), get_device()
    if "--h168" in sys.argv:
        h168(P)
    if "--seeds" in sys.argv:
        seeds(P, device)
    if "--seeds-nf" in sys.argv:
        seeds_nf(P)
    if "--local" in sys.argv:
        local_vs_global(P, device)
    if "--retrain-nf" in sys.argv:
        retrain_nf(P)
    if "--retrain-lgbm" in sys.argv:
        retrain_lgbm(P)
    if "--retrain-mstl" in sys.argv:
        retrain_mstl(P)
