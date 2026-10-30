"""Etapa 6: Chronos-Bolt zero-shot (sin entrenamiento) con cuantiles P10/P50/P90.

Se evalúa en validación (línea de referencia de la curva de historia) y en prueba."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import backtest                   # noqa: E402
from src.library_models import ChronosForecaster      # noqa: E402
from src.train_torch import Panel                     # noqa: E402
from src.utils import save_run                        # noqa: E402

if __name__ == "__main__":
    P = Panel()
    tic = time.perf_counter()
    fc = ChronosForecaster("amazon/chronos-bolt-base", ctx=2048)
    load = time.perf_counter() - tic
    meta = dict(family="Fundacional", L=2048, strategy="multi-salida (cuantiles)",
                covariates="No", params_total=fc.n_params, params_trainable=0,
                train_time=0.0, load_time=load)
    from src.utils import list_runs
    for split, origins in (("val", P.val_origins), ("test", P.test_origins)):
        if "Chronos-Bolt" in list_runs(split):
            continue                                  # ya evaluado
        res = backtest(fc, P.Y, origins, scale=P.scale)
        save_run(res, dict(meta), split=split)
        m = res["metrics"]
        print(f"[{split}] Chronos-Bolt MASE {m['MASE']:.4f} pinball {m['pinball']:.2f} "
              f"cov80 {m['coverage_80']:.3f} infer {res['infer_time']:.1f}s", flush=True)
    # H = 168 (orígenes diarios cuyo horizonte semanal cabe en prueba)
    o168 = P.test_origins[P.test_origins + 168 <= P.t_test_end + 1]
    fc.name = "Chronos-Bolt_H168"
    fc.forecast = lambda hist, t0: fc._run(hist, t0, H=168)[0][..., 1]
    fc.quantiles = None
    res = backtest(fc, P.Y, o168, H=168, scale=P.scale)
    save_run(res, dict(H=168), split="test168")
    print(f"Chronos-Bolt H=168 MASE {res['metrics']['MASE']:.4f}")
