"""Protocolo único de evaluación: backtesting con origen móvil diario + métricas en kWh."""
import time

import numpy as np
import pandas as pd

from . import config as C


def daily_origins(index, start, end, H=C.H):
    """Posiciones enteras de los orígenes diarios (00:00) cuyo horizonte cae en [start, end]."""
    idx = pd.DatetimeIndex(index)
    t_lo, t_hi = idx.get_loc(pd.Timestamp(start)), idx.get_loc(pd.Timestamp(end))
    return np.arange(t_lo, t_hi - H + 2, H)


def mase_scale(Y, t_train_end, m=C.SEASON):
    """Escala de MASE por cliente: MAE in-sample del seasonal naive (m=24) en entrenamiento."""
    tr = Y[:t_train_end + 1]
    return np.abs(tr[m:] - tr[:-m]).mean(axis=0)            # (N,)


class Forecaster:
    """Interfaz común. `forecast(hist, t0)` recibe SOLO Y[:t0] (kWh, forma (t0, N)) y
    devuelve (N, H) — o (N, H, Q) en `forecast_quantiles` para modelos probabilísticos."""
    name = "base"
    quantiles = None

    def forecast(self, hist, t0):
        raise NotImplementedError


class Precomputed(Forecaster):
    """Adaptador para librerías que ya hacen su propio backtest sin fuga
    (cross_validation de Nixtla con refit=False): busca el pronóstico por origen."""

    def __init__(self, name, by_origin, q_by_origin=None, quantiles=None):
        self.name, self.by_origin = name, by_origin
        self.q_by_origin, self.quantiles = q_by_origin, quantiles

    def forecast(self, hist, t0):
        return self.by_origin[t0]

    def forecast_quantiles(self, hist, t0):
        return self.q_by_origin[t0]


def backtest(model, Y, origins, H=C.H, scale=None):
    """Ejecuta el backtest de `model` sobre `origins`.

    En cada origen t0 el modelo solo ve Y[:t0] (datos hasta las 23:00 del día
    anterior) y pronostica Y[t0:t0+H]. Devuelve dict con F (N, O, H), A, quantiles
    opcionales y el tiempo de inferencia total.
    """
    F, Q = [], []
    tic = time.perf_counter()
    for t0 in origins:
        hist = Y[:t0]                                    # corte duro: nada del futuro
        F.append(model.forecast(hist, t0))
        if getattr(model, "quantiles", None) is not None:
            Q.append(model.forecast_quantiles(hist, t0))
    infer_time = time.perf_counter() - tic
    F = np.stack(F, axis=1).astype("float64")            # (N, O, H)
    A = np.stack([Y[t0:t0 + H].T for t0 in origins], axis=1).astype("float64")
    out = dict(name=model.name, F=F, A=A, origins=np.asarray(origins),
               infer_time=infer_time)
    if Q:
        out["Q"] = np.stack(Q, axis=1)                   # (N, O, H, nq)
        out["quantiles"] = model.quantiles
    if scale is not None:
        out["metrics"] = metrics(F, A, scale, out.get("Q"), out.get("quantiles"))
    return out


def pinball(Q, A, qs):
    """Pinball loss promedio sobre cuantiles: max(q*e, (q-1)*e) con e = y - q_hat."""
    losses = []
    for j, q in enumerate(qs):
        e = A - Q[..., j]
        losses.append(np.maximum(q * e, (q - 1) * e).mean())
    return float(np.mean(losses))


def metrics(F, A, scale, Q=None, qs=None, mask_origins=None):
    """Métricas en kWh agregadas sobre todos los clientes y orígenes."""
    if mask_origins is not None:
        F, A = F[:, mask_origins], A[:, mask_origins]
        Q = None if Q is None else Q[:, mask_origins]
    E = F - A
    mae_c = np.abs(E).mean(axis=(1, 2))
    rmse_c = np.sqrt((E ** 2).mean(axis=(1, 2)))
    nz = A > 0
    ape = np.abs(E[nz]) / A[nz]
    m = dict(
        MAE=float(np.abs(E).mean()),
        RMSE=float(np.sqrt((E ** 2).mean())),
        WAPE=float(np.abs(E).sum() / np.abs(A).sum()),
        MAPE=float(ape.mean()),                 # excluye y=0 (división indefinida)
        MAPE_p99=float(np.quantile(ape, 0.99)),
        MASE=float((mae_c / scale).mean()),
        MAE_by_client=mae_c, RMSE_by_client=rmse_c, MASE_by_client=mae_c / scale,
        MAE_by_h=np.abs(E).mean(axis=(0, 1)),
        MASE_by_h=(np.abs(E).mean(axis=1) / scale[:, None]).mean(axis=0),
    )
    if Q is not None:
        lo, hi = qs.index(0.1), qs.index(0.9)
        m["pinball"] = pinball(Q, A, qs)
        m["coverage_80"] = float(((A >= Q[..., lo]) & (A <= Q[..., hi])).mean())
    return m


def by_hour_of_day(F, A, origins, index):
    """MAE por hora del día (0..23) según el timestamp de cada pronóstico."""
    hours = np.array([[pd.Timestamp(index[t0 + h]).hour for h in range(F.shape[2])]
                      for t0 in origins])                         # (O, H)
    err = np.abs(F - A).mean(axis=0)                              # (O, H)
    return pd.Series(err.ravel()).groupby(hours.ravel()).mean()


def xmas_mask(origins, index):
    d = pd.DatetimeIndex([index[t] for t in origins])
    return (d >= pd.Timestamp(C.XMAS_START)) & (d <= pd.Timestamp(C.XMAS_END))


def summary_row(res, scale, index):
    """Fila resumen (H=24) con métricas globales, MASE de Navidad y probabilísticas."""
    m = metrics(res["F"], res["A"], scale, res.get("Q"), res.get("quantiles"))
    mx = metrics(res["F"], res["A"], scale, mask_origins=xmas_mask(res["origins"], index))
    row = dict(model=res["name"], MAE=m["MAE"], RMSE=m["RMSE"], WAPE=m["WAPE"],
               MAPE=m["MAPE"], MASE=m["MASE"], MASE_xmas=mx["MASE"],
               infer_s=res["infer_time"])
    if "pinball" in m:
        row.update(pinball=m["pinball"], coverage_80=m["coverage_80"])
    return row


def to_long_forecasts(res, index, columns):
    """Pronósticos en formato (unique_id, ds, modelo, y_hat)."""
    N, O, H = res["F"].shape
    ds = np.array([[index[t0 + h] for h in range(H)] for t0 in res["origins"]])  # (O,H)
    return pd.DataFrame(dict(
        unique_id=np.repeat(np.asarray(columns), O * H),
        ds=np.tile(ds.ravel(), N),
        modelo=res["name"],
        y_hat=res["F"].reshape(N, -1).ravel().astype("float32")))
