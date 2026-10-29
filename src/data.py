"""Etapas de datos: crudo 15 min -> serie horaria limpia -> panel -> ventanas -> tensores."""
import urllib.request
import zipfile

import holidays
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C


# ----------------------------------------------------------------------------
# 1. Descarga y carga del crudo
# ----------------------------------------------------------------------------
def download_raw():
    """Descarga el zip de UCI y extrae LD2011_2014.txt en data/raw/ (si no existe)."""
    if C.RAW_TXT.exists():
        return C.RAW_TXT
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = C.RAW_DIR / "electricity.zip"
    if not zpath.exists():
        urllib.request.urlretrieve(C.RAW_URL, zpath)
    zipfile.ZipFile(zpath).extract("LD2011_2014.txt", C.RAW_DIR)
    return C.RAW_TXT


def load_raw():
    """Matriz cruda: 140,256 filas (cada 15 min) x 370 clientes, en kW."""
    download_raw()
    raw = pd.read_csv(C.RAW_TXT, sep=";", decimal=",", index_col=0, parse_dates=True)
    raw.index.name = "ds"
    return raw.astype("float32")


# ----------------------------------------------------------------------------
# 2. Agregación horaria y cambios de horario
# ----------------------------------------------------------------------------
def to_hourly(raw):
    """kW cada 15 min -> kWh por hora (kWh = sum(kW * 0.25 h)).

    Cada hora agrupa los 4 registros etiquetados HH:00, HH:15, HH:30 y HH:45 y queda
    etiquetada por su inicio. Se eligió esta alineación porque el bloque anómalo de
    los días de cambio de horario ocupa exactamente las etiquetas 01:00-01:45; con
    ella la anomalía cae completa en la hora 01:00 y no contamina horas vecinas.
    Horas incompletas (bordes 2011-01-01 00:00 y 2015-01-01 00:00) quedan como NaN.
    """
    return raw.resample("h").sum(min_count=4) * 0.25


def dst_days(years=(2011, 2012, 2013, 2014)):
    """Último domingo de marzo (23 h) y de octubre (25 h) de cada año (horario de Portugal)."""
    out = []
    for y in years:
        for m in (3, 10):
            last = pd.Timestamp(y, m, 31)
            out.append((m, last - pd.Timedelta(days=(last.weekday() + 1) % 7)))
    return out


def fix_dst(hourly):
    """Corrige los dos días de cambio de horario.

    El dataset guarda 96 registros en todos los días:
    - marzo (día de 23 h): la hora 01:00 viene en cero para todos los clientes
      -> se imputa con el promedio de las horas 00:00 y 02:00;
    - octubre (día de 25 h): la hora 01:00 acumula dos horas reales
      -> se divide entre 2 (se reparte el consumo).
    Así la serie queda en una rejilla regular de 24 h/día sin huecos ni picos falsos.
    """
    h = hourly.copy()
    for month, day in dst_days():
        t = day + pd.Timedelta(hours=1)
        if t not in h.index:
            continue
        if month == 3:
            prev, nxt = t - pd.Timedelta(hours=1), t + pd.Timedelta(hours=1)
            h.loc[t] = (h.loc[prev] + h.loc[nxt]) / 2
        else:
            h.loc[t] = h.loc[t] / 2
    return h


def complete_clients(hourly, start=C.START, end=C.TEST_END, max_zero_run=24):
    """Clientes con datos completos en [start, end]: el crudo rellena con ceros antes
    del alta (y después de una baja), así que se exige que ninguna racha de ceros
    consecutivos dentro del periodo supere `max_zero_run` horas. Esto descarta a
    quienes empiezan después del 2012-01-01, terminan antes de 2015 o tienen apagones
    largos de medición."""
    per = hourly.loc[start:end]
    zero = (per == 0)
    runs = zero.apply(lambda s: s.groupby((~s).cumsum()).sum().max())
    ok = per.notna().all() & (runs <= max_zero_run)
    return ok[ok].index.tolist()


def build_hourly(force=False):
    """Crudo -> horario limpio 2012-2014 (todos los clientes válidos) -> parquet."""
    if C.HOURLY_ALL.exists() and not force:
        return pd.read_parquet(C.HOURLY_ALL)
    raw = load_raw()
    hourly = fix_dst(to_hourly(raw))
    keep = complete_clients(hourly)
    out = hourly.loc[C.START:C.TEST_END, keep].astype("float32")
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.HOURLY_ALL)
    return out


def build_panel(force=False):
    """Subconjunto fijo de N_CLIENTS clientes (semilla 42), igual para todos los modelos."""
    if C.HOURLY_PANEL.exists() and not force:
        return pd.read_parquet(C.HOURLY_PANEL)
    allc = build_hourly()
    rng = np.random.default_rng(C.SEED)
    cols = sorted(rng.choice(allc.columns.to_numpy(), size=C.N_CLIENTS, replace=False))
    panel = allc[cols]
    panel.to_parquet(C.HOURLY_PANEL)
    return panel


def to_long(panel):
    """Formato largo de Nixtla: (unique_id, ds, y)."""
    df = panel.reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
    return df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. Covariables de calendario (conocidas a futuro)
# ----------------------------------------------------------------------------
def pt_holidays(years=(2011, 2012, 2013, 2014)):
    return holidays.country_holidays("PT", years=list(years))


def calendar_features(index):
    """Codificación seno/coseno de hora (24), día de semana (7) y mes (12) + bandera
    de feriado de Portugal. Todas dependen solo del timestamp, por lo que se conocen
    en el instante de emitir el pronóstico para cualquier hora futura."""
    idx = pd.DatetimeIndex(index)
    hol = pt_holidays()
    f = pd.DataFrame(index=idx)
    for name, val, period in (("hour", idx.hour, 24), ("dow", idx.dayofweek, 7),
                              ("month", idx.month - 1, 12)):
        f[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
    f["holiday"] = np.array([d in hol for d in idx.date], dtype="float32")
    f["weekend"] = (idx.dayofweek >= 5).astype("float32")
    return f.astype("float32")


N_CAL = 8


# ----------------------------------------------------------------------------
# 4. Normalización por serie con estadísticos SOLO de entrenamiento
# ----------------------------------------------------------------------------
class SeriesScaler:
    """z-score por cliente. fit() recibe únicamente el bloque de entrenamiento."""

    def fit(self, train_values):            # (T_train, N)
        self.mu = train_values.mean(axis=0)
        self.sd = train_values.std(axis=0) + 1e-6
        return self

    def transform(self, v):
        return (v - self.mu) / self.sd

    def inverse(self, v, idx=None):
        mu, sd = (self.mu, self.sd) if idx is None else (self.mu[idx], self.sd[idx])
        return v * sd + mu


def time_index(panel, ts):
    """Posición entera del timestamp `ts` en el índice horario del panel."""
    return int(panel.index.get_loc(pd.Timestamp(ts)))


# ----------------------------------------------------------------------------
# 5. Dataset de ventanas deslizantes
# ----------------------------------------------------------------------------
class WindowDataset(Dataset):
    """Ventanas deslizantes sobre TODAS las series.

    Para cada serie i y cada origen t (primera hora a pronosticar):
      X      = [y_norm, calendario][t-L : t]       -> (L, 1 + N_CAL)
      X_fut  = calendario[t : t+H]                 -> (H, N_CAL)   (conocido a futuro)
      y      = y_norm[t : t+H]                     -> (H,)
    Los orígenes válidos se limitan a [t_lo, t_hi] de modo que el OBJETIVO completo
    caiga dentro del bloque (train/val/test) y la entrada solo use el pasado.
    """

    def __init__(self, values_norm, cal, t_lo, t_hi, L=C.L, H=C.H, stride=1,
                 series=None):
        self.v = torch.as_tensor(values_norm, dtype=torch.float32)   # (T, N)
        self.cal = torch.as_tensor(cal, dtype=torch.float32)         # (T, N_CAL)
        self.L, self.H = L, H
        t_lo = max(t_lo, L)                                          # se necesita historia L
        self.origins = np.arange(t_lo, t_hi - H + 2, stride)
        self.series = np.arange(self.v.shape[1]) if series is None else np.asarray(series)

    def __len__(self):
        return len(self.origins) * len(self.series)

    def __getitem__(self, k):
        i = self.series[k % len(self.series)]
        t = int(self.origins[k // len(self.series)])
        y_in = self.v[t - self.L:t, i].unsqueeze(-1)
        X = torch.cat([y_in, self.cal[t - self.L:t]], dim=-1)
        return X, self.cal[t:t + self.H], self.v[t:t + self.H, i], i
