"""Modelos de librería adaptados a la interfaz común de backtest():
StatsForecast (baselines), LightGBM global, NeuralForecast y Chronos-Bolt."""
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import calendar_features
from .evaluation import Forecaster, Precomputed

CAL_COLS = list(calendar_features(pd.DatetimeIndex(["2014-01-01"])).columns)


def long_df(P, t_end, t_start=0, with_cal=False, cols=None, values=None):
    """Panel -> formato largo (unique_id, ds, y[, calendario]) con filas t_start..t_end-1.
    `values` permite pasar la matriz ya normalizada (por defecto, kWh)."""
    cols = np.arange(P.N) if cols is None else np.asarray(cols)
    V = P.Y if values is None else values
    idx = P.index[t_start:t_end]
    df = pd.DataFrame(dict(
        unique_id=np.repeat(P.columns[cols].to_numpy(), len(idx)),
        ds=np.tile(idx.to_numpy(), len(cols)),
        y=V[t_start:t_end, cols].T.ravel()))
    if with_cal:
        cal = np.tile(P.cal[t_start:t_end], (len(cols), 1))
        df[CAL_COLS] = cal
    return df


def _pivot_cv(cv, P, origins, col, cols=None):
    """cross_validation de Nixtla -> {t0: (N, H)} en el orden del panel."""
    names = P.columns if cols is None else P.columns[np.asarray(cols)]
    out = {}
    cv = cv.copy()
    cv["t0"] = P.index.get_indexer(cv["cutoff"] + pd.Timedelta(hours=1))
    for t0, g in cv.groupby("t0"):
        piv = g.pivot(index="unique_id", columns="ds", values=col).loc[names]
        out[int(t0)] = piv.to_numpy()
    assert set(out) == set(int(t) for t in origins)
    return out


# ----------------------------------------------------------------------------
# StatsForecast
# ----------------------------------------------------------------------------
def stats_models():
    from statsforecast.models import MSTL, AutoETS, Naive, SeasonalNaive
    return [Naive(), SeasonalNaive(24, alias="SeasonalNaive24"),
            SeasonalNaive(168, alias="SeasonalNaive168"),
            MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"),
                 alias="MSTL")]


def stats_backtest(P, origins, models=None, H=C.H, input_size=24 * 7 * 8):
    """StatsForecast.cross_validation con n_windows = #orígenes, step_size = 24 y
    refit=False: cada modelo se ajusta una sola vez con datos hasta el primer corte y
    luego solo avanza (`forward`) con los datos nuevos en cada origen. `input_size`
    limita a las últimas 8 semanas la historia usada para ajustar (MSTL es local)."""
    from statsforecast import StatsForecast
    models = models or stats_models()
    t_end = int(origins[-1]) + H
    df = long_df(P, t_end)
    sf = StatsForecast(models=models, freq="h", n_jobs=1)   # el pool recompila numba
    tic = time.perf_counter()
    cv = sf.cross_validation(h=H, df=df, n_windows=len(origins), step_size=H,
                             refit=False, input_size=input_size)
    elapsed = time.perf_counter() - tic
    out = {}
    for m in models:
        name = m.alias if hasattr(m, "alias") else type(m).__name__
        out[name] = Precomputed(name, _pivot_cv(cv, P, origins, name))
    return out, elapsed


# ----------------------------------------------------------------------------
# LightGBM global recursivo
# ----------------------------------------------------------------------------
LAGS = list(range(1, 25)) + [168]


class LGBMForecaster(Forecaster):
    """LightGBM global (un solo modelo para todos los clientes) con rezagos 1-24 y 168,
    calendario (hora, día, mes, feriado) e id del cliente como categórica. El objetivo se
    normaliza por serie (z-score de entrenamiento). Multi-paso recursivo: en el paso h
    los rezagos < h se llenan con las propias predicciones."""

    def __init__(self, P, months=30, params=None, seed=C.SEED, name="LightGBM",
                 rounds=2000):
        self.P, self.months, self.name, self.rounds = P, months, name, rounds
        self.params = dict(objective="l1", learning_rate=0.05, num_leaves=127,
                           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                           bagging_freq=1, verbose=-1, seed=seed, num_threads=12)
        self.params.update(params or {})

    def _cal_int(self, t):
        idx = self.P.index[t]
        hol = self.P.cal[t, CAL_COLS.index("holiday")]
        return np.stack([idx.hour, idx.dayofweek, idx.month, hol], axis=-1)

    def _features(self, Z, t, cols):
        """Filas para (series `cols`, tiempo t) usando solo Z[t - lag]."""
        lags = np.stack([Z[t - l, cols] for l in LAGS], axis=-1)
        cal = np.repeat(self._cal_int(np.array([t]))[None], len(cols), axis=0).reshape(len(cols), -1)
        return np.concatenate([lags, cal, cols[:, None]], axis=1)

    def _matrix(self, Z, t_lo, t_hi):
        """Matriz de entrenamiento vectorizada para t in [t_lo, t_hi]."""
        ts = np.arange(t_lo, t_hi + 1)
        N = Z.shape[1]
        lag = np.stack([Z[ts - l] for l in LAGS], axis=-1)              # (T, N, nl)
        cal = np.repeat(self._cal_int(ts)[:, None], N, axis=1)          # (T, N, 4)
        sid = np.broadcast_to(np.arange(N)[None, :, None], (len(ts), N, 1))
        X = np.concatenate([lag, cal, sid], axis=-1).reshape(-1, len(LAGS) + 5)
        return X, Z[ts].reshape(-1)

    def fit(self, t_train_end=None, t_val=None):
        import lightgbm as lgb
        P = self.P
        t_train_end = P.t_train_end if t_train_end is None else t_train_end
        t_lo = max(168, t_train_end + 1 - self.months * 30 * 24)
        tr = P.Y[t_lo:t_train_end + 1]
        self.mu, self.sd = tr.mean(0), tr.std(0) + 1e-6       # stats solo de entrenamiento
        Z = (P.Y - self.mu) / self.sd
        names = [f"lag{l}" for l in LAGS] + ["hour", "dow", "month", "holiday", "client"]
        Xtr, ytr = self._matrix(Z, t_lo, t_train_end)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=["client"])
        valid, cb = [], []
        if t_val is not None:                                   # early stopping en validación
            Xv, yv = self._matrix(Z, t_val[0], t_val[1])
            valid = [lgb.Dataset(Xv, yv, reference=dtr)]
            cb = [lgb.early_stopping(50, verbose=False)]
        tic = time.perf_counter()
        self.booster = lgb.train(self.params, dtr, num_boost_round=self.rounds,
                                 valid_sets=valid, callbacks=cb)
        self.train_time = time.perf_counter() - tic
        self.n_params = self.booster.num_trees()
        return self

    def forecast(self, hist, t0, H=C.H):
        N = hist.shape[1]
        cols = np.arange(N)
        Zh = (hist[-200:] - self.mu) / self.sd                     # solo el pasado
        buf = np.concatenate([Zh, np.zeros((H, N))], axis=0)
        base = len(Zh)
        for h in range(H):
            t = base + h
            lags = np.stack([buf[t - l] for l in LAGS], axis=-1)
            cal = np.repeat(self._cal_int(np.array([t0 + h])), N, axis=0)
            X = np.concatenate([lags, cal, cols[:, None]], axis=1)
            buf[t] = self.booster.predict(X, num_threads=12)
        return (buf[base:] * self.sd + self.mu).T


# ----------------------------------------------------------------------------
# NeuralForecast
# ----------------------------------------------------------------------------
def train_stats(P, t_end=None):
    """Media y desviación por cliente SOLO con el bloque de entrenamiento."""
    t_end = P.t_train_end if t_end is None else t_end
    tr = P.Y[:t_end + 1]
    return tr.mean(0), tr.std(0) + 1e-6


class NFForecaster(Forecaster):
    """Envuelve un NeuralForecast ya entrenado: en cada origen arma el df largo con las
    últimas `ctx` horas de hist (solo pasado, z-score por serie con estadísticos de
    entrenamiento) y el futr_df de calendario; luego regresa a kWh.

    Nota: se normaliza fuera de NeuralForecast porque su `scaler_type='standard'` también
    escala las exógenas y la bandera de feriado (constante = 0 en casi todas las ventanas,
    desviación 0) hace explotar las predicciones."""

    def __init__(self, nf, P, alias, mu, sd, ctx=24 * 21, quantiles=None, cols=None):
        self.nf, self.P, self.name, self.ctx = nf, P, alias, ctx
        self.mu, self.sd, self.quantiles = mu, sd, quantiles
        self.cols = np.arange(P.N) if cols is None else np.asarray(cols)
        self._cache = {}

    def _predict(self, hist, t0):
        if t0 in self._cache:
            return self._cache[t0]
        P, H, c = self.P, C.H, self.cols
        Zh = np.zeros((t0 + H, P.N), dtype="float32")
        Zh[t0 - self.ctx:t0] = (hist[t0 - self.ctx:t0] - self.mu) / self.sd   # solo pasado
        df = long_df(P, t0, t0 - self.ctx, with_cal=True, cols=c, values=Zh)
        futr = long_df(P, t0 + H, t0, with_cal=True, cols=c, values=Zh).drop(columns="y")
        kw = dict(quantiles=self.quantiles) if self.quantiles else {}
        pred = self.nf.predict(df=df, futr_df=futr, **kw)
        self._cache = {t0: pred}
        return pred

    def _pivot(self, pred, col):
        names = self.P.columns[self.cols]
        z = pred.pivot(index="unique_id", columns="ds", values=col).loc[names].to_numpy()
        return z * self.sd[self.cols, None] + self.mu[self.cols, None]        # a kWh

    def _qcols(self, pred):
        """{cuantil: columna}; NeuralForecast usa 'Modelo_ql0.1' o 'Modelo-ql0.1'."""
        import re
        out = {}
        for c in pred.columns:
            m = re.fullmatch(re.escape(self.name) + r"[_-]ql([0-9.]+)", c)
            if m:
                out[float(m.group(1))] = c
        return out

    def forecast(self, hist, t0):
        pred = self._predict(hist, t0)
        q = self._qcols(pred)
        col = q[0.5] if 0.5 in q else self.name      # probabilísticos: la mediana
        return self._pivot(pred, col)

    def forecast_quantiles(self, hist, t0):
        pred = self._predict(hist, t0)
        qcol = self._qcols(pred)
        return np.stack([self._pivot(pred, qcol[q]) for q in self.quantiles], axis=-1)


def nf_fit(models, P, t_fit_end, val_size=None, t_start=0, cols=None, stats=None):
    """Entrena NeuralForecast con datos [t_start, t_fit_end] (inclusive), en z-score por
    serie. Los últimos `val_size` pasos (bloque de validación) son para early stopping."""
    from neuralforecast import NeuralForecast
    mu, sd = stats if stats is not None else train_stats(P)
    Z = ((P.Y - mu) / sd).astype("float32")
    # el calendario solo se incluye si algún modelo lo usa: NeuralForecast arma todas las
    # ventanas de cada lote de series con todas las columnas y la memoria crece en proporción
    with_cal = any(getattr(m, "futr_exog_list", None) for m in models)
    df = long_df(P, t_fit_end + 1, t_start, with_cal=with_cal, cols=cols, values=Z)
    nf = NeuralForecast(models=models, freq="h")
    tic = time.perf_counter()
    nf.fit(df=df, val_size=val_size or 0)
    return nf, time.perf_counter() - tic


# ----------------------------------------------------------------------------
# Chronos-Bolt (zero-shot)
# ----------------------------------------------------------------------------
class ChronosForecaster(Forecaster):
    """Chronos-Bolt sin entrenamiento: contexto = últimas `ctx` horas de cada cliente."""

    def __init__(self, model_id="amazon/chronos-bolt-base", ctx=2048, device=None,
                 quantiles=C.QUANTILES, name="Chronos-Bolt", batch=25):
        from chronos import BaseChronosPipeline
        dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=dev,
                                                        torch_dtype=torch.float32)
        self.ctx, self.quantiles, self.qlev, self.name = ctx, quantiles, list(quantiles), name
        self.batch = batch
        self.n_params = sum(p.numel() for p in self.pipe.model.parameters())
        self._last = None

    def _run(self, hist, t0, H=C.H):
        if self._last is not None and self._last[0] == t0 and self._last[1] == H:
            return self._last[2]
        ctx = torch.as_tensor(hist[-self.ctx:].T.copy(), dtype=torch.float32)
        qs, ms = [], []
        for k in range(0, len(ctx), self.batch):          # lotes acotan la memoria (MPS)
            q, mean = self.pipe.predict_quantiles(ctx[k:k + self.batch], prediction_length=H,
                                                  quantile_levels=self.qlev)
            qs.append(q.cpu().numpy()); ms.append(mean.cpu().numpy())
        self._last = (t0, H, (np.concatenate(qs), np.concatenate(ms)))
        return self._last[2]

    def forecast(self, hist, t0):
        q, _ = self._run(hist, t0)
        return q[..., self.qlev.index(0.5)]                      # mediana como punto

    def forecast_quantiles(self, hist, t0):
        return self._run(hist, t0)[0]
