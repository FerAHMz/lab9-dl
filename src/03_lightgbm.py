"""Etapa 3: LightGBM global con rezagos 1-24 y 168 + calendario.

Iteración de historia (3, 12, 30 meses) evaluada SOLO en validación; la mejor se
evalúa una vez en prueba."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgbm  # noqa: E402,F401  (antes que torch: evita choque de OpenMP en macOS)
from src.evaluation import backtest                    # noqa: E402
from src.library_models import LGBMForecaster          # noqa: E402
from src.train_torch import Panel                      # noqa: E402
from src.utils import log_iteration, save_run          # noqa: E402

if __name__ == "__main__":
    P = Panel()
    val_rng = (P.t_val, P.t_val_end)
    best = None
    for months in (3, 12, 30):
        m = LGBMForecaster(P, months=months, name=f"LightGBM_m{months}")
        m.fit(t_val=val_rng)
        res = backtest(m, P.Y, P.val_origins, scale=P.scale)
        mase = res["metrics"]["MASE"]
        print(f"LightGBM {months:2d} meses: val MASE {mase:.4f}  árboles {m.n_params} "
              f"({m.train_time:.0f}s)", flush=True)
        meta = dict(family="Árboles", L="rezagos 1-24,168", strategy="recursiva",
                    covariates="Sí", params_total=m.n_params, params_trainable=m.n_params,
                    train_time=m.train_time, months=months)
        save_run(res, meta, split="val")
        log_iteration(dict(model="LightGBM", tag=f"LightGBM_m{months}", L=168, H=24,
                           months=months, best_val_MASE=mase, val_MAE=res["metrics"]["MAE"],
                           params_total=m.n_params, train_time=m.train_time,
                           infer_time_val=res["infer_time"]), group="history")
        if best is None or mase < best[0]:
            best = (mase, months, m)
    _, months, m = best
    m.name = "LightGBM"
    res = backtest(m, P.Y, P.test_origins, scale=P.scale)
    save_run(res, dict(family="Árboles", L="rezagos 1-24,168", strategy="recursiva",
                       covariates="Sí", params_total=m.n_params, params_trainable=m.n_params,
                       train_time=m.train_time, months=months))
    print(f"LightGBM prueba ({months} meses): MASE {res['metrics']['MASE']:.4f}")
