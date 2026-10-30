"""Etapa 5: modelos de NeuralForecast (N-HiTS, TCN, PatchTST, iTransformer, DeepAR).

Cada modelo se entrena UNA vez con datos hasta 2014-09-30; los últimos 2208 pasos
(jul-sep) son el bloque de validación para early stopping. Luego backtest() con los 92
orígenes de prueba (cada predicción usa solo las 3 semanas previas al origen).
PatchTST además se entrena con 3, 12 y 30 meses de historia (evaluado en validación).
"""
import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from neuralforecast.losses.pytorch import MAE, DistributionLoss   # noqa: E402
from neuralforecast.models import (NHITS, TCN, DeepAR, PatchTST,  # noqa: E402
                                   iTransformer)

from src import config as C                                        # noqa: E402
from src.evaluation import backtest                                # noqa: E402
from src.library_models import CAL_COLS, NFForecaster, nf_fit, train_stats      # noqa: E402
from src.train_torch import Panel                                  # noqa: E402
from src.utils import log_iteration, save_run                      # noqa: E402

VAL = 92 * 24
TRAINER = dict(accelerator="cpu", enable_progress_bar=False, logger=False,
               enable_checkpointing=False, enable_model_summary=False)
# inference_windows_batch_size: por defecto (-1) valida TODAS las ventanas en un lote y la
# memoria se dispara; con lotes de 512 se mantiene acotada.
COMMON = dict(h=C.H, max_steps=1500, val_check_steps=100, early_stop_patience_steps=4,
              random_seed=C.SEED, inference_windows_batch_size=512, **TRAINER)


def build(name, L=168, seed=C.SEED, max_steps=1500):
    kw = dict(COMMON, input_size=L, random_seed=seed, max_steps=max_steps)
    if name == "DeepAR":
        # La validación interna de DeepAR muestrea 100 trayectorias para ~218k ventanas y
        # agota la RAM: se entrena con pasos fijos y sin early stopping (no toca la prueba).
        kw.update(max_steps=2500, early_stop_patience_steps=-1)
    if name == "NHITS":
        return NHITS(**kw, futr_exog_list=CAL_COLS, scaler_type="identity", loss=MAE(),
                     n_pool_kernel_size=[8, 4, 1], n_freq_downsample=[24, 12, 1],
                     windows_batch_size=512, batch_size=64, learning_rate=1e-3)
    if name == "TCN":
        return TCN(**kw, futr_exog_list=CAL_COLS, scaler_type="identity", loss=MAE(),
                   kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 128],
                   encoder_hidden_size=128, context_size=24, decoder_hidden_size=128,
                   windows_batch_size=256, batch_size=16, learning_rate=1e-3)
    if name == "PatchTST":
        return PatchTST(**kw, scaler_type="identity", revin=True, loss=MAE(),
                        patch_len=24, stride=12, hidden_size=128, n_heads=8,
                        encoder_layers=3, linear_hidden_size=256, dropout=0.2,
                        windows_batch_size=512, batch_size=64, learning_rate=1e-4)
    if name == "iTransformer":
        return iTransformer(**kw, n_series=C.N_CLIENTS, scaler_type="identity", loss=MAE(),
                            hidden_size=256, n_heads=8, e_layers=2, d_ff=512, dropout=0.1,
                            batch_size=32, learning_rate=1e-3)
    if name == "DeepAR":
        return DeepAR(**kw, futr_exog_list=CAL_COLS, scaler_type="identity",
                      loss=DistributionLoss("StudentT", level=[80]), valid_loss=MAE(),
                      lstm_hidden_size=128, lstm_n_layers=2, trajectory_samples=100,
                      windows_batch_size=128, batch_size=16, learning_rate=1e-3)
    raise ValueError(name)


FAMILY = {"NHITS": ("Residuales y conv.", "multi-salida", "Sí"),
          "TCN": ("Residuales y conv.", "multi-salida (MLP decoder)", "Sí"),
          "PatchTST": ("Atención", "multi-salida", "No"),
          "iTransformer": ("Atención", "multi-salida", "No"),
          "DeepAR": ("Probabilísticos", "autorregresiva (muestreo)", "Sí")}


def count(nf):
    m = nf.models[0]
    return (sum(p.numel() for p in m.parameters()),
            sum(p.numel() for p in m.parameters() if p.requires_grad))


def fit_eval(name, P, t_fit_end, origins, split, t_start=0, seed=C.SEED, alias=None,
             meta_extra=None, max_steps=1500):
    model = build(name, seed=seed, max_steps=max_steps)
    if alias:
        model.alias = alias
    stats = train_stats(P)                       # z-score con estadísticos de entrenamiento
    val = 0 if name == "DeepAR" else VAL
    nf, train_time = nf_fit([model], P, t_fit_end, val_size=val, t_start=t_start, stats=stats)
    alias = alias or name
    try:                                         # respaldo opcional del modelo entrenado
        nf.save(str(C.RESULTS_DIR / "ckpt" / f"nf_{alias}"), overwrite=True)
    except Exception as e:
        print(f"  (no se pudo guardar {alias}: {type(e).__name__})", flush=True)
    q = C.QUANTILES if name == "DeepAR" else None
    fc = NFForecaster(nf, P, alias, *stats, quantiles=q)
    res = backtest(fc, P.Y, origins, scale=P.scale)
    n_tot, n_tr = count(nf)
    fam, strat, cov = FAMILY[name]
    meta = dict(family=fam, L=168, strategy=strat, covariates=cov, params_total=n_tot,
                params_trainable=n_tr, train_time=train_time,
                steps=int(nf.models[0].trainer.global_step) if hasattr(nf.models[0], "trainer") else None,
                **(meta_extra or {}))
    save_run(res, meta, split=split)
    print(f"[{split}] {alias:22s} MASE {res['metrics']['MASE']:.4f} train {train_time:.0f}s "
          f"infer {res['infer_time']:.0f}s params {n_tot:,}", flush=True)
    return res, meta


if __name__ == "__main__":
    P = Panel()
    which = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = which or ([] if "--history" in sys.argv else list(FAMILY))
    for name in which:
        try:
            # DeepAR (recurrente) solo con los últimos 24 meses para acotar la memoria
            t0 = P.t_test - 1 - 2 * 365 * 24 if name == "DeepAR" else 0
            fit_eval(name, P, P.t_test - 1, P.test_origins, "test", t_start=t0)
        except Exception as e:                       # un fallo no detiene a los demás
            print(f"ERROR {name}: {type(e).__name__}: {e}", flush=True)
    if "--history" in sys.argv:
        from src.utils import list_runs
        done = set(list_runs("val"))
        # historia variable: entrenamiento termina el 2014-06-30, validación jul-sep
        for months in [m for m in (3, 12, 30) if f"PatchTST_m{m}" not in done]:
            t_start = max(0, P.t_train_end + 1 - months * 30 * 24 - 168)
            res, meta = fit_eval("PatchTST", P, P.t_val_end, P.val_origins, "val",
                                 t_start=t_start, alias=f"PatchTST_m{months}",
                                 meta_extra=dict(months=months), max_steps=500)
            log_iteration(dict(model="PatchTST", tag=f"PatchTST_m{months}", L=168, H=24,
                               months=months, best_val_MASE=res["metrics"]["MASE"],
                               params_total=meta["params_total"],
                               train_time=meta["train_time"]), group="history")
