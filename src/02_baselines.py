"""Etapa 2: baselines de StatsForecast (Naive, SeasonalNaive 24/168, MSTL) con backtest
de 92 orígenes en prueba, y también sobre validación (para la curva de historia)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import backtest                        # noqa: E402
from src.library_models import stats_backtest              # noqa: E402
from src.train_torch import Panel                          # noqa: E402
from src.utils import save_run                             # noqa: E402

if __name__ == "__main__":
    P = Panel()
    for split, origins in (("val", P.val_origins), ("test", P.test_origins)):
        tic = time.perf_counter()
        fcs, elapsed = stats_backtest(P, origins)
        print(f"[{split}] StatsForecast cross_validation: {elapsed:.1f}s")
        for name, fc in fcs.items():
            res = backtest(fc, P.Y, origins, scale=P.scale)
            # en cross_validation ajuste e inferencia van juntos: se reporta todo como
            # inferencia y entrenamiento ~ 0 salvo MSTL (ajuste local por serie).
            res["infer_time"] = elapsed / len(fcs)
            L = {"Naive": "1 (último valor)", "SeasonalNaive24": "24",
                 "SeasonalNaive168": "168", "MSTL": "8 semanas"}[name]
            meta = dict(family="Baseline", L=L,
                        strategy="directa" if "Naive" in name else "multi-salida",
                        covariates="No", params_total=0, params_trainable=0,
                        train_time=0.0 if "Naive" in name else elapsed / len(fcs))
            save_run(res, meta, split=split)
            print(f"  {name:18s} MASE {res['metrics']['MASE']:.4f}  MAE {res['metrics']['MAE']:.2f}")
