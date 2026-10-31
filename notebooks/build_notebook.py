"""Genera notebooks/lab9_series_tiempo.ipynb a partir de celdas definidas aquí.

Uso:  python notebooks/build_notebook.py && jupyter nbconvert --to notebook --execute \
      --inplace notebooks/lab9_series_tiempo.ipynb
"""
from pathlib import Path

import nbformat as nbf

CELLS = []


def md(s):
    CELLS.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(nbf.v4.new_code_cell(s.strip("\n")))


# =============================================================================
md(r"""
# Laboratorio 9 — Series de tiempo y forecasting
**CC3092 Deep Learning y Sistemas Inteligentes — UVG**

Fernando Andree Hernández Martínez (23645) · Fernando José Rueda Rodas (23748)

Pronóstico *day-ahead* (H = 24 h) del consumo eléctrico horario de clientes de Portugal
(ElectricityLoadDiagrams20112014, UCI). El notebook sigue el pipeline completo:
datos crudos → series horarias limpias → ventanas → tensores → modelos → `backtest()` →
comparación. Los entrenamientos largos se ejecutan con los scripts de `src/` (mismas
funciones que se importan aquí) y sus resultados se guardan en `results/`; este notebook
los carga, los audita y los visualiza.

| Etapa | Script |
|---|---|
| Datos crudos → panel horario | `src/01_prepare_data.py` |
| Baselines StatsForecast | `src/02_baselines.py` |
| LightGBM global | `src/03_lightgbm.py` |
| Modelos propios PyTorch + iteraciones | `src/04_torch_models.py` |
| NeuralForecast (N-HiTS, TCN, PatchTST, iTransformer, DeepAR) | `src/05_neuralforecast.py` |
| Chronos-Bolt zero-shot | `src/06_chronos.py` |
| Reentrenamiento mensual, local vs global, semillas | `src/07_extras.py` |
""")

code(r"""
import sys, inspect, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from IPython.display import Code, display

from src import config as C
from src import data as D
from src.evaluation import (backtest, metrics, mase_scale, daily_origins, summary_row,
                            by_hour_of_day, xmas_mask, to_long_forecasts, Forecaster)
from src.train_torch import Panel, TrainConfig, train_model, get_device
from src import torch_models as TM
from src.utils import load_run, list_runs, ITER_CSV, EPOCH_CSV

SEED = C.SEED                     # reproducibilidad
np.random.seed(SEED); torch.manual_seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
FIG = C.FIG_DIR; FIG.mkdir(parents=True, exist_ok=True)
show = lambda obj: display(Code(inspect.getsource(obj), language="python"))
print("torch", torch.__version__, "| device:", get_device())
""")

# =============================================================================
md(r"""
## 1. Datos

El crudo trae 140,256 registros cada 15 min (2011–2014) para 370 clientes, en kW.
La variable objetivo es el **consumo por hora en kWh**: kWh = Σ kW·0.25 h de los 4 cuartos.
""")
code(r"""
raw = D.load_raw()
print("crudo:", raw.shape, "|", raw.index[0], "->", raw.index[-1])
raw.iloc[:6, :5]
""")
code(r"""
show(D.to_hourly)
hourly0 = D.to_hourly(raw)
hourly0.loc["2012-01-01"].iloc[:4, :5]
""")

md(r"""
### 1.1 Cambios de horario

Todos los días del crudo tienen 96 registros, aun los de 23 h y 25 h. Inspeccionando el
total del panel alrededor de la 01:00 del último domingo de marzo y de octubre:
""")
code(r"""
s = raw.sum(axis=1) / 1e3
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
for a, d, t in ((ax[0], "2013-03-31", "marzo (día de 23 h)"), (ax[1], "2013-10-27", "octubre (día de 25 h)")):
    s.loc[f"{d} 00:00":f"{d} 03:45"].plot(ax=a, marker="o", ms=3)
    a.set_title(f"{d}: {t}"); a.set_ylabel("MW (suma 370 clientes)")
plt.tight_layout(); plt.savefig(FIG / "dst.png", bbox_inches="tight"); plt.show()
""")
md(r"""
- **Marzo**: los registros 01:00–01:45 vienen en cero (esa hora no existió). Se **imputa**
  la hora 01:00 con el promedio de las horas 00:00 y 02:00 de cada cliente.
- **Octubre**: los registros 01:00–01:45 suman dos horas reales (≈ doble). Se **divide
  entre 2**.

Así cada serie queda en una rejilla regular de 24 h/día (necesaria para ventanas, lags 24/168
y la estacionalidad de MASE) sin un cero falso ni un pico falso que los modelos aprenderían
como patrón. El bloque anómalo ocupa exactamente las etiquetas 01:00–01:45, por eso cada
hora agrupa las etiquetas HH:00–HH:45.
""")
code(r"""
show(D.fix_dst)
hourly = D.fix_dst(hourly0)
for m, d in D.dst_days((2012, 2013, 2014)):
    t = d + pd.Timedelta(hours=1)
    print(d.date(), "01:00 antes:", round(hourly0.loc[t].sum()/1e3, 1), "MWh  -> después:",
          round(hourly.loc[t].sum()/1e3, 1), " | 00:00 =", round(hourly.loc[t - pd.Timedelta(hours=1)].sum()/1e3, 1))
""")

md(r"""
### 1.2 Filtro del periodo 2012–2014 y subconjunto de clientes

Antes del alta de un cliente (o después de su baja) el crudo rellena con ceros. Un cliente
tiene **datos completos desde 2012-01-01** si dentro de 2012-01-01 → 2014-12-31 no tiene
ninguna racha de ceros de más de 24 h.
""")
code(r"""
show(D.complete_clients)
allc = D.build_hourly()
panel = D.build_panel()
print(f"Clientes tras el filtro 2012–2014: {allc.shape[1]} de {raw.shape[1]}")
print(f"Observaciones horarias por serie: {allc.shape[0]:,}  (1,096 días x 24 h)")
print(f"Panel completo filtrado: {allc.size:,} observaciones")
print(f"Subconjunto usado (semilla {C.SEED}): {panel.shape[1]} clientes -> {panel.size:,} observaciones")
""")
md(r"""
Por cómputo (≈20 modelos × 92 orígenes en una laptop) se usa el subconjunto permitido de
**100 clientes elegidos con semilla 42**, idéntico para todos los modelos.
""")

# =============================================================================
md("## 2. Exploración")
code(r"""
P = Panel()
Y, idx = P.Y, P.index
hol = D.pt_holidays()
norm_load = panel / panel.loc[:C.TRAIN_END].mean()          # cada cliente relativo a su media
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
prof_h = norm_load.groupby(norm_load.index.hour).mean()
ax[0].plot(prof_h.index, prof_h.median(axis=1), lw=2, label="mediana clientes")
ax[0].fill_between(prof_h.index, prof_h.quantile(.25, axis=1), prof_h.quantile(.75, axis=1), alpha=.3, label="IQR")
ax[0].set(title="Perfil por hora del día", xlabel="hora", ylabel="consumo / media del cliente"); ax[0].legend()
dow = norm_load.groupby(norm_load.index.dayofweek).mean().median(axis=1)
ax[1].bar(["L","M","X","J","V","S","D"], dow); ax[1].set(title="Perfil por día de la semana")
hm = norm_load.median(axis=1).groupby([norm_load.index.dayofweek, norm_load.index.hour]).mean().unstack()
im = ax[2].imshow(hm, aspect="auto", cmap="viridis"); ax[2].set(title="Hora x día (mediana)", xlabel="hora")
ax[2].set_yticks(range(7), ["L","M","X","J","V","S","D"]); plt.colorbar(im, ax=ax[2])
plt.tight_layout(); plt.savefig(FIG / "profiles.png", bbox_inches="tight"); plt.show()
""")
code(r"""
daily = panel.sum(axis=1).resample("D").sum() / 1e3
hdays = [pd.Timestamp(d) for d in hol if pd.Timestamp("2012-01-01") <= pd.Timestamp(d) <= pd.Timestamp("2014-12-31")]
fig, ax = plt.subplots(2, 1, figsize=(13, 6))
ax[0].plot(daily.index, daily, lw=.8)
ax[0].scatter(hdays, daily.loc[hdays], color="crimson", s=14, zorder=3, label="feriado PT")
for a, b in ((C.VAL_START, C.VAL_END), (C.TEST_START, C.TEST_END)):
    ax[0].axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=.12)
ax[0].set(title="Consumo diario del panel (MWh) con feriados de Portugal; sombreado = validación / prueba"); ax[0].legend()
for y in (2012, 2013, 2014):
    w = daily.loc[f"{y}-12-10":f"{y+1}-01-10"] if y < 2014 else daily.loc[f"{y}-12-10":f"{y}-12-31"]
    ax[1].plot((w.index - pd.Timestamp(f"{y}-12-25")).days, w / w.iloc[:7].mean(), marker=".", label=str(y))
ax[1].axvline(0, color="k", ls="--"); ax[1].axvline(7, color="k", ls=":")
ax[1].set(title="Navidad (0) y Año Nuevo (7): consumo relativo a la 2a semana de diciembre", xlabel="días desde el 25-dic"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG / "holidays.png", bbox_inches="tight"); plt.show()
xm = {y: (daily.loc[f"{y}-12-25"] / daily.loc[f"{y}-12-10":f"{y}-12-16"].mean() - 1) for y in (2012, 2013, 2014)}
ny = {y: (daily.loc[f"{y+1}-01-01"] / daily.loc[f"{y}-12-10":f"{y}-12-16"].mean() - 1) for y in (2012, 2013)}
print("Cambio el 25-dic vs. 2a semana de dic:", {k: f"{v:+.1%}" for k, v in xm.items()})
print("Cambio el 01-ene vs. 2a semana de dic:", {k: f"{v:+.1%}" for k, v in ny.items()})
""")
md(r"""
**Lectura.** Doble estacionalidad marcada: diaria (valle 2–6 h, meseta 9–19 h y pico
nocturno de iluminación) y semanal (sábado y sobre todo domingo más bajos). Los feriados se
comportan como domingos. En **Navidad y Año Nuevo** el consumo cae con fuerza (ver cifras)
y la semana entre ambos queda por debajo de lo normal: es un cambio de régimen corto que un
seasonal naive de 168 h no puede anticipar (copia la semana previa, que fue normal).
""")
code(r"""
mean_c = panel.loc[:C.TRAIN_END].mean()
print("Consumo promedio por cliente (kWh/h, entrenamiento):")
print(mean_c.describe()[["min", "50%", "max"]].round(2).to_string())
print(f"razón max/min = {mean_c.max()/mean_c.min():.0f}x")
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.hist(np.log10(mean_c), bins=25); ax.set(xlabel="log10 consumo medio (kWh/h)", ylabel="clientes", title="Escalas entre clientes")
plt.tight_layout(); plt.savefig(FIG / "scales.png", bbox_inches="tight"); plt.show()
""")
md(r"""
**Normalización.** Las escalas difieren en ~3 órdenes de magnitud. Con una normalización
**global** la pérdida la dominan los 2–3 clientes grandes y los pequeños quedan en ≈0
(se verifica en la iteración de normalización). Elegimos **z-score por serie con media y
desviación del bloque de entrenamiento**: pone a todos los clientes en la misma escala para
un modelo global, es invertible para reportar en kWh y no usa información del futuro. Como
alternativa se probó la normalización **por ventana** (RevIN sin parámetros), que además
corrige cambios de nivel; PatchTST y los modelos de NeuralForecast la usan internamente.
""")

md(r"""
### 2.1 Covariables de calendario conocidas a futuro

Hora (24), día de semana (7) y mes (12) se codifican con **seno/coseno**: dos columnas
continuas por variable que preservan la circularidad (23 h está junto a 0 h, diciembre junto
a enero) sin las 43 columnas de un one-hot. *Feriado* (librería `holidays`, Portugal) y
*fin de semana* son banderas binarias. Solo dependen del timestamp, por lo que se conocen
para cualquier hora futura en el instante del pronóstico.
""")
code(r"""
show(D.calendar_features)
cal = D.calendar_features(idx)
cal.loc["2014-12-24 22:00":"2014-12-25 02:00"].round(3)
""")

md(r"""
### 2.2 Dataset de ventanas deslizantes (PyTorch)

`WindowDataset` recorre todas las series y todos los orígenes válidos con stride 1 y
devuelve `X: (L, 1+8)` (consumo normalizado + calendario del pasado), `X_fut: (H, 8)`
(calendario del horizonte) y `y: (H,)`. Los orígenes de cada bloque se restringen para que
el **objetivo completo** caiga dentro del bloque; la entrada puede venir del bloque anterior
(es pasado observado en ese instante).
""")
code(r"""
show(D.WindowDataset)
scaler = D.SeriesScaler().fit(Y[:P.t_train_end + 1])            # solo entrenamiento
Z, calv = scaler.transform(Y), cal.to_numpy()
ds_tr = D.WindowDataset(Z, calv, 0, P.t_train_end)
ds_va = D.WindowDataset(Z, calv, P.t_val, P.t_val_end)
ds_te = D.WindowDataset(Z, calv, P.t_test, P.t_test_end)
for n, d in (("entrenamiento", ds_tr), ("validación", ds_va), ("prueba", ds_te)):
    print(f"{n:14s}: {len(d.origins):6,} orígenes x {len(d.series)} series = {len(d):,} ventanas")
loader = torch.utils.data.DataLoader(ds_tr, batch_size=32, shuffle=True)
X, Xf, yb, ib = next(iter(loader))
print("X", tuple(X.shape), "| X_fut", tuple(Xf.shape), "| y", tuple(yb.shape))
""")
md(r"""
Para el backtest se usan además los **92 orígenes diarios** (00:00) de validación y de
prueba (`daily_origins`), que son el subconjunto de estas ventanas con stride 24.
""")

md(r"""
### 2.3 Auditoría de fugas

Regla: cada entrada de la red debe poder calcularse en el instante en que se emite el
pronóstico (00:00 del día *d*, con datos hasta las 23:00 del día *d−1*).
""")
code(r"""
# (1) Split aleatorio de ventanas -> NO: los bloques son cronológicos y disjuntos en el OBJETIVO.
last_tr_target = idx[ds_tr.origins.max() + C.H - 1]
first_va_target, last_va_target = idx[ds_va.origins.min()], idx[ds_va.origins.max() + C.H - 1]
first_te_target = idx[ds_te.origins.min()]
print("último objetivo de train:", last_tr_target, "| primer objetivo de val:", first_va_target)
print("último objetivo de val:  ", last_va_target, "| primer objetivo de test:", first_te_target)
assert last_tr_target < first_va_target and last_va_target < first_te_target
# El barajado (shuffle) ocurre SOLO dentro de los orígenes de entrenamiento:
src = inspect.getsource(train_model)
print([l.strip() for l in src.splitlines() if "tr_origins = " in l or "rng.choice(tr_origins" in l])
""")
code(r"""
# (2) Escalar con toda la serie -> NO: mu/sd se ajustan solo con el bloque de entrenamiento.
sc_full = D.SeriesScaler().fit(Y)                       # lo que NO hacemos
print("diferencia media(train) vs media(serie completa), mediana entre clientes:",
      f"{np.median(np.abs(scaler.mu - sc_full.mu) / scaler.sd):.3f} desv.")
assert np.allclose(scaler.mu, Y[:P.t_train_end + 1].mean(0))
print([l.strip() for l in src.splitlines() if "Normalizer(cfg.norm" in l])
""")
code(r"""
# (3) Early stopping con la prueba -> NO: el criterio es el MASE de los 92 orígenes de VALIDACIÓN.
print("orígenes de validación:", idx[P.val_origins[0]], "->", idx[P.val_origins[-1]])
print("el último horizonte de validación termina en", idx[P.val_origins[-1] + C.H - 1], "<", idx[P.t_test])
assert P.val_origins.max() + C.H - 1 < P.t_test
print([l.strip() for l in src.splitlines() if "val_origins = " in l or "best_state, wait = val_mase" in l])
""")
code(r"""
# (4) Covariables mal alineadas -> NO: X termina a las 23:00 del día anterior, X_fut empieza a las 00:00
# del día pronosticado y solo contiene calendario (función del timestamp).
t0 = P.test_origins[84]                                   # 2014-12-24
print("entrada X:", idx[t0 - C.L], "->", idx[t0 - 1], "| calendario futuro:", idx[t0], "->", idx[t0 + C.H - 1])
x, xf, yy, _ = ds_te[list(ds_te.origins).index(t0) * len(ds_te.series)]
assert np.allclose(xf.numpy(), cal.iloc[t0:t0 + C.H].to_numpy())
print("feriado en X_fut (25-dic al día siguiente):", cal.iloc[t0 + 24:t0 + 26]["holiday"].tolist(), "| en X_fut[0] (24-dic):", xf[0, -2].item())
""")
code(r"""
# Prueba empírica para TODOS los forecasters: si se alteran los valores futuros (>= t0),
# el pronóstico emitido en t0 no cambia. backtest() solo entrega Y[:t0] al modelo.
show(backtest)
""")

# =============================================================================
md(r"""
## 3. Investigación: herramientas para forecasting en PyTorch

**`nn.LSTM` / `nn.GRU`** — redes recurrentes con compuertas. `input_size`: features por paso;
`hidden_size`: dimensión del estado; `num_layers`: capas apiladas; `batch_first=True` usa
entradas `(B, L, F)`; `dropout`: entre capas (solo si `num_layers>1`). Salidas:
`output (B, L, hidden)` = estado oculto de la última capa en cada paso; `h_n (layers, B, hidden)`
= último estado de cada capa; `c_n` (solo LSTM) = última celda de memoria. La GRU no tiene
`c_n` (fusiona celda y estado) y tiene ~25 % menos parámetros.

**`nn.Conv1d` con `dilation` y padding causal** — `Conv1d(in, out, kernel_size, dilation=d)`;
para que la salida en *t* no vea *t+1…*, se rellena **solo a la izquierda** con
`(k−1)·d` ceros (o `padding=(k−1)·d` y se recortan los últimos elementos). Una TCN apila
bloques con dilataciones 1, 2, 4, …; el campo receptivo es
`R = 1 + Σ_b (k−1)·d_b` (× 2 si cada bloque residual tiene dos convoluciones). Con k=2 y
d = 1…128 (nuestra TCN): R = 1 + 255 = 256 h > L = 168.

**`nn.TransformerEncoderLayer`** — `d_model`: dimensión de los tokens, `nhead`: cabezas de
atención (d_model divisible por nhead), `dim_feedforward`: ancho del MLP interno,
`batch_first`, `dropout`. La atención es invariante a permutaciones, así que se **suma** una
codificación posicional (senoidal fija o embedding aprendido `nn.Embedding(L, d_model)`) al
embedding de entrada antes del encoder. PatchTST tokeniza parches de 24 h (canal
independiente); iTransformer usa cada **serie completa** como token y aplica la atención
entre variables.

**NeuralForecast / StatsForecast (Nixtla)** — entrada en formato largo
`(unique_id, ds, y)` más columnas exógenas. `cross_validation(df, h, n_windows, step_size,
refit)` hace backtesting con origen móvil: `n_windows` cortes, separados `step_size` pasos
(nosotros 92 y 24); `refit=False` entrena una sola vez (con el primer corte) y solo predice
en los siguientes, `refit=True`/entero reentrena en cada/cada *k* ventanas. En NeuralForecast
`val_size` reserva el final del entrenamiento para early stopping.

**Chronos-Bolt `BaseChronosPipeline.predict_quantiles`** — `inputs`: tensor `(B, T)` o lista
de tensores 1-D con el contexto (hasta 2048 pasos en Bolt); `prediction_length`: horizonte
(Bolt predice 64 pasos de una vez, más allá rueda autorregresivamente); `quantile_levels`:
lista de cuantiles (Bolt fue entrenado con 0.1…0.9). Devuelve `(quantiles (B, H, Q), mean (B, H))`.

**Modelo local vs. global** — local: un modelo por serie (solo ve su historia; no comparte
patrones, costo × N). Global: un modelo con todas las series (más datos, transfiere
patrones comunes, requiere normalización por serie).

**Teacher forcing** — al entrenar un modelo autorregresivo/seq2seq se alimenta el valor
**real** anterior en vez de la predicción. Acelera y estabiliza el entrenamiento, pero crea
*exposure bias*: en inferencia el modelo ve sus propios errores, que se acumulan con *h*.

**RevIN** — normaliza cada ventana de entrada con su media y desviación (más un afín
aprendible) y desnormaliza la salida con los mismos estadísticos. Ante cambios de nivel
(una serie que sube o baja de régimen) el modelo ve siempre entradas centradas y la
desnormalización traslada el nivel reciente al pronóstico.

**Pinball loss** — `ρ_q(y, ŷ) = max(q(y−ŷ), (q−1)(y−ŷ))`. Su minimizador es el cuantil *q*
de la distribución condicional; promediada sobre varios *q* aproxima el CRPS. La cobertura
P10–P90 debería ser ≈ 80 %.

**MASE vs WAPE** — ambas libres de escala. **MASE** = MAE / MAE in-sample del seasonal
naive (m = 24) **por serie** y luego se promedia: cada cliente pesa igual y < 1 significa
mejor que el naive estacional. **WAPE** = Σ|e| / Σ|y| sobre todo el panel: pondera por
volumen (los clientes grandes dominan) y se lee como % del consumo total.
""")

# =============================================================================
md(r"""
## 4. Protocolo de validación y métricas

Una sola función `backtest(modelo)` para todos los modelos. En cada uno de los 92 orígenes
(00:00 de cada día de oct–dic 2014) el modelo recibe **solo `Y[:t0]`** y devuelve las 24
horas siguientes. Los pesos se entrenan una vez con datos hasta 2014-09-30 (entrenamiento +
validación para early stopping). Los modelos de librería se envuelven con la misma interfaz
`Forecaster.forecast(hist, t0)`.
""")
code(r"""
show(Forecaster); show(metrics)
""")
code(r"""
# Ejemplo: seasonal naive en numpy pasado por backtest() y la prueba de "futuro alterado"
class SNaive(Forecaster):
    def __init__(self, m): self.m, self.name = m, f"SNaive{m}"
    def forecast(self, hist, t0): return hist[-self.m:][:C.H].T    # mismo día de hace m horas

def future_invariance(model, t0):
    Yp = Y.copy(); Yp[t0:] = Yp[t0:] * 3 + 1000           # alterar el futuro
    return np.allclose(model.forecast(Y[:t0], t0), model.forecast(Yp[:t0], t0))

sn = SNaive(168)
r = backtest(sn, Y, P.test_origins, scale=P.scale)
print({k: round(v, 4) for k, v in r["metrics"].items() if isinstance(v, float)})
print("invariante al futuro:", future_invariance(sn, P.test_origins[10]))
""")
md(r"""
**MAPE y clientes de consumo cercano a cero.** El MAPE divide entre el valor real: con
horas en 0 kWh es indefinido (se excluyen) y con horas de 0.1–1 kWh un error pequeño en kWh
se vuelve un error de cientos o miles por ciento. Por eso el promedio lo dominan pocos
clientes y horas nocturnas; abajo se muestra la distribución del APE por cliente. Para
comparar modelos usamos MASE (por cliente) y WAPE (por volumen).
""")
code(r"""
E = np.abs(r["F"] - r["A"]); A_ = r["A"]
ape_c = np.array([np.mean(E[i][A_[i] > 0] / A_[i][A_[i] > 0]) for i in range(P.N)])
low = np.argsort(P.Y[:P.t_train_end].mean(0))[:5]
print("MAPE por cliente (SNaive168): mediana", f"{np.median(ape_c):.1%}", "| máx", f"{ape_c.max():.0%}")
print("5 clientes de menor consumo:", {P.columns[i]: f"{ape_c[i]:.0%}" for i in low})
print("fracción de horas reales < 1 kWh en prueba:", f"{(A_ < 1).mean():.2%}")
""")

# =============================================================================
md(r"""
## 5. Construcción y entrenamiento de los modelos

### 5.0 Implementaciones propias (PyTorch)
Todas reciben `X (B, L, 1+8)` y `X_fut (B, H, 8)` y devuelven `(B, 24)` en escala normalizada.
""")
code(r"""
for cls in (TM.MLP, TM.DLinear, TM.NLinear, TM.LSTMMimo, TM.LSTMAutoregressive, TM.LSTMSeq2Seq):
    show(cls)
""")
code(r"""
from src import train_torch as TT
show(TT.train_model)
""")
md(r"""
Los modelos de librería (StatsForecast, LightGBM, NeuralForecast y Chronos) se envuelven con la
misma interfaz `Forecaster` en `src/library_models.py`:
""")
code(r"""
from src import library_models as LM
show(LM.LGBMForecaster); show(LM.NFForecaster.forecast); show(LM.ChronosForecaster)
""")

md("### 5.1 Estrategias multi-paso con la LSTM")
code(r"""
it = pd.read_csv(ITER_CSV); ep = pd.read_csv(EPOCH_CSV)
strat = it[(it.group == "long") & it.model.str.startswith("LSTM")].copy()   # mismo presupuesto (40 épocas)
strat["estrategia"] = strat.apply(lambda r: {"LSTM-AR": "autorregresiva", "LSTM-MIMO": "multi-salida",
    "LSTM-Seq2Seq": "encoder-decoder" + (" (realimenta y, TF 0.5)" if "_fp" in r.tag else "")}[r.model], axis=1)
display(strat[["estrategia", "best_val_MASE", "params_total", "epochs_run", "train_time"]].round(4))
""")
code(r"""
# MAE por paso del horizonte de cada estrategia (en VALIDACIÓN: la prueba se usa una vez por modelo)
from importlib import import_module
tm04 = import_module("src.04_torch_models") if False else None
import importlib.util
spec = importlib.util.spec_from_file_location("s04", ROOT / "src" / "04_torch_models.py")
s04 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s04)
device = get_device()
fig, ax = plt.subplots(figsize=(8, 3.4))
for _, row in strat.iterrows():
    fc, cfg = s04.load_forecaster(row, P, device, row.estrategia)
    rv = backtest(fc, Y, P.val_origins, scale=P.scale)
    ax.plot(range(1, 25), rv["metrics"]["MAE_by_h"], marker=".", label=f"{row.estrategia} (MASE {rv['metrics']['MASE']:.3f})")
ax.set(xlabel="paso del horizonte h", ylabel="MAE (kWh)", title="LSTM: MAE por paso según estrategia (validación)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "strategies.png", bbox_inches="tight"); plt.show()
""")

md("### 5.2 Iteraciones de hiperparámetros (solo validación)")
code(r"""
cols = ["group", "model", "L", "H", "months", "norm", "hidden", "layers", "dropout", "lr", "batch",
        "params_total", "params_trainable", "best_epoch", "epochs_run", "best_val_MASE",
        "epoch_time_mean", "train_time", "peak_mem_MB"]
pd.set_option("display.max_rows", 80)
it[[c for c in cols if c in it]].round(4)
""")
code(r"""
# Curvas de pérdida (train vs. validación) y MASE de validación por epoch
sel = [t for t in it.tag if t.startswith(("DLinear_L336_H24_m30_series", "LSTM-Seq2Seq_L168_H24_m30_series_h128_l2_d0.1_lr0.001_b512",
                                           "MLP_L168_H24_m30_series_h128_l2_d0.1_lr0.001_b512", "LSTM-AR_L168"))][:4]
fig, ax = plt.subplots(2, len(sel), figsize=(3.6 * len(sel), 5.2), sharex="col")
for j, t in enumerate(sel):
    e = ep[ep.tag == t]
    ax[0, j].plot(e.epoch, e.train_loss, label="train"); ax[0, j].plot(e.epoch, e.val_loss, label="val")
    ax[0, j].set_title(t.split("_m30")[0], fontsize=9); ax[0, j].legend(fontsize=7)
    ax[1, j].plot(e.epoch, e.val_MASE, color="C2"); ax[1, j].set_xlabel("epoch")
ax[0, 0].set_ylabel("pérdida L1 (norm.)"); ax[1, 0].set_ylabel("MASE val")
plt.tight_layout(); plt.savefig(FIG / "loss_curves.png", bbox_inches="tight"); plt.show()
""")
code(r"""
# ¿Menor pérdida de entrenamiento => mejor MASE de validación?
last = ep.groupby("tag").agg(train_loss=("train_loss", "last"), val_MASE=("val_MASE", "min")).join(it.set_index("tag")[["model", "norm"]])
last = last[last.norm == "series"]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for m, g in last.groupby("model"):
    ax.scatter(g.train_loss, g.val_MASE, label=m, s=18)
ax.set(xlabel="pérdida final de entrenamiento", ylabel="mejor MASE validación"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
print("correlación de Spearman:", round(last[["train_loss", "val_MASE"]].corr("spearman").iloc[0, 1], 3))
""")

md("### 5.3 Modelo local contra global")
code(r"""
lg = pd.read_csv(C.RESULTS_DIR / "local_vs_global.csv")
best_name = load_run(pd.read_csv(C.RESULTS_DIR / "seeds.csv").model.iloc[0]).get("name")
g = load_run(best_name)
mae_c = np.abs(g["F"] - g["A"]).mean(axis=(1, 2))
lg["global_MASE"] = mae_c[lg.idx] / P.scale[lg.idx]
lg["mejora_global"] = 1 - lg.global_MASE / lg.local_MASE
display(lg[["client", "mean_kwh", "local_MASE", "global_MASE", "mejora_global"]].round(3))
print("MASE promedio 10 clientes  local:", round(lg.local_MASE.mean(), 4), "| global:", round(lg.global_MASE.mean(), 4))
""")

# =============================================================================
md("## 6. Comparación de resultados (prueba, 92 orígenes)")
code(r"""
runs = list_runs("test")
rows = []
for n in runs:
    r = load_run(n); m = r["meta"]
    row = summary_row(r, P.scale, idx)
    row.update(family=m.get("family"), L=m.get("L"), strategy=m.get("strategy"), covariates=m.get("covariates"),
               params_total=m.get("params_total"), params_trainable=m.get("params_trainable"),
               train_s=m.get("train_time"), hardware=m.get("hardware"))
    rows.append(row)
h168 = {}
for n in list_runs("test168"):
    r = load_run(n, "test168"); h168[n.replace("_H168", "")] = metrics(r["F"], r["A"], P.scale)["MASE"]
table = pd.DataFrame(rows)
table["MASE_H168"] = table.model.map(h168)
table = table.sort_values("MASE").reset_index(drop=True)
order = ["model", "family", "L", "strategy", "covariates", "params_total", "params_trainable", "train_s", "infer_s",
         "MAE", "RMSE", "WAPE", "MAPE", "MASE", "MASE_H168", "MASE_xmas", "pinball", "coverage_80", "hardware"]
table = table[[c for c in order if c in table]]
table.to_csv(C.RESULTS_DIR / "comparison_table.csv", index=False)
table.round(4)
""")
code(r"""
# MAE y RMSE por cliente (kWh): distribución entre los 100 clientes para cada modelo
pc = {}
for n in table.model:
    r = load_run(n); m = metrics(r["F"], r["A"], P.scale)
    pc[(n, "MAE")], pc[(n, "RMSE")] = m["MAE_by_client"], m["RMSE_by_client"]
per_client = pd.DataFrame(pc, index=P.columns)
per_client.to_csv(C.RESULTS_DIR / "metrics_by_client.csv")
per_client.describe().loc[["min", "25%", "50%", "75%", "max"]].T.unstack().round(1).iloc[:, :10]
""")
code(r"""
# (1) Frontera de Pareto: MASE vs. tiempo de entrenamiento (log). Chronos (sin entrenamiento) y el
# naive (MASE 4) se muestran como referencia para no aplastar la escala.
t = table[~table.model.isin(["Chronos-Bolt", "Naive"])].copy()
t["train_plot"] = t.train_s.fillna(0).clip(lower=0.5)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.scatter(t.train_plot, t.MASE, s=28, zorder=3)
for _, r in t.iterrows():
    ax.annotate(r.model, (r.train_plot, r.MASE), fontsize=7, xytext=(3, 3), textcoords="offset points")
ts = t.sort_values("train_plot"); front = ts[ts.MASE <= ts.MASE.cummin()]
ax.step(front.train_plot, front.MASE, where="post", color="crimson", lw=1.3, label="frontera de Pareto")
ax.axhline(table.set_index("model").loc["Chronos-Bolt", "MASE"], ls="--", color="k", lw=1, label="Chronos-Bolt zero-shot (0 s)")
ax.set(xscale="log", xlabel="tiempo de entrenamiento (s, log; sin entrenamiento -> 0.5 s)", ylabel="MASE prueba", ylim=(0.7, 1.4))
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "pareto.png", bbox_inches="tight"); plt.show()
print("frontera:", " -> ".join(f"{r.model} ({r.MASE:.3f})" for _, r in front.iterrows()))
""")
code(r"""
# (2) MAE por paso del horizonte, una curva por modelo
fig, ax = plt.subplots(figsize=(9, 4.6))
colors, styles = plt.cm.tab20(np.linspace(0, 1, 20)), ["-", "--", "-.", ":"]
for k, n in enumerate(table.model):
    if n == "Naive": continue
    r = load_run(n); m = metrics(r["F"], r["A"], P.scale)
    ax.plot(range(1, 25), m["MAE_by_h"], lw=1.4, color=colors[k % 20], ls=styles[k % 4], label=n)
ax.set(xlabel="h (horas desde el origen 00:00)", ylabel="MAE (kWh)", title="MAE por paso del horizonte (prueba)")
ax.legend(fontsize=7, ncol=3); plt.tight_layout(); plt.savefig(FIG / "mae_by_h.png", bbox_inches="tight"); plt.show()
""")
md(r"""
Como todos los orígenes son a las 00:00, el paso *h* coincide con la hora del día *h−1*: el
error por hora del día es la misma curva desplazada (se calcula explícitamente abajo). La
forma de la curva mezcla dos efectos: degradación con el horizonte y dificultad propia de
cada hora (las horas de arranque de actividad, 7–10 h, son las más difíciles).
""")
code(r"""
hod = pd.DataFrame({n: by_hour_of_day(load_run(n)["F"], load_run(n)["A"], load_run(n)["origins"], idx)
                    for n in table.model[:6]})
hod.index.name = "hora"; hod.round(2).T
""")
code(r"""
# (3) MASE vs. meses de historia (validación) con Chronos como referencia
itv = pd.read_csv(ITER_CSV)
base30 = ["DLinear_L168_H24_m30_series_h128_l2_d0.1_lr0.001_b512_s42",            # 30 meses (corrida base)
          "LSTM-Seq2Seq_L168_H24_m30_series_h128_l2_d0.1_lr0.001_b512_s42"]
hist = itv[(itv.group == "history") | itv.tag.isin(base30)]
fig, ax = plt.subplots(figsize=(7, 3.8))
for m, g in hist.groupby("model"):
    g = g.sort_values("months"); ax.plot(g.months, g.best_val_MASE, marker="o", label=m)
for n, ls in (("SeasonalNaive24", ":"), ("Chronos-Bolt", "--")):
    try:
        rv = load_run(n, "val"); ax.axhline(metrics(rv["F"], rv["A"], P.scale)["MASE"], ls=ls, color="gray" if "Naive" in n else "k", label=f"{n} (no depende de la historia)")
    except FileNotFoundError: pass
ax.set(xlabel="meses de historia de entrenamiento", ylabel="MASE validación", xticks=[3, 12, 30]); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "history.png", bbox_inches="tight"); plt.show()
""")

md("### 6.1 Resultados probabilísticos")
code(r"""
prob = table.dropna(subset=["pinball"])[["model", "MAE", "MASE", "pinball", "coverage_80"]]
display(prob.round(4))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
i = int(np.argsort(P.Y[:P.t_train_end].mean(0))[P.N // 2])        # cliente mediano
for a, n in zip(axs, prob.model):
    r = load_run(n); o = slice(80, 88)                               # 20-27 dic
    tt = np.concatenate([np.arange(t0, t0 + 24) for t0 in r["origins"][o]])
    a.plot(idx[tt], r["A"][i, o].ravel(), "k", lw=1, label="real")
    a.plot(idx[tt], r["Q"][i, o, :, 1].ravel(), lw=1, label="P50")
    a.fill_between(idx[tt], r["Q"][i, o, :, 0].ravel(), r["Q"][i, o, :, 2].ravel(), alpha=.3, label="P10–P90")
    a.set_title(f"{n} — {P.columns[i]}"); a.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "probabilistic.png", bbox_inches="tight"); plt.show()
""")

md(r"""### 6.2 Semana del 22 al 28 de diciembre, reentrenamiento mensual y semillas

Reentrenamiento: iTransformer (mejor DL entrenado) y LightGBM se reentrenan el 1 de oct, nov y
dic con toda la historia disponible (ventana expandible). MSTL no se repite: StatsForecast ya lo
reajusta en cada origen aun con `refit=False` (su `forward` re-descompone la serie), por lo que el
resultado sería idéntico.""")
code(r"""
xm = table[["model", "MASE", "MASE_xmas"]].assign(deterioro=lambda d: d.MASE_xmas / d.MASE - 1)
display(xm.round(3))
""")
code(r"""
rt = []
for n in list_runs("retrain"):
    r = load_run(n, "retrain"); base = r["meta"]["base"]
    once = table.set_index("model").loc[base, "MASE"] if base in set(table.model) else np.nan
    rt.append(dict(modelo=base, MASE_una_vez=once, MASE_reentreno_mensual=metrics(r["F"], r["A"], P.scale)["MASE"]))
pd.DataFrame(rt).round(4)
""")
code(r"""
for f in ("seeds.csv", "seeds_itransformer.csv"):
    sd = pd.read_csv(C.RESULTS_DIR / f)
    print(f"{sd.model.iloc[0]:13s} MASE prueba por semilla:", dict(zip(sd.seed, sd.test_MASE.round(4))),
          f"-> {sd.test_MASE.mean():.4f} ± {sd.test_MASE.std(ddof=1):.4f}")
""")

md("### 6.3 Archivo de pronósticos de prueba")
code(r"""
fc_long = pd.concat([to_long_forecasts(load_run(n), idx, P.columns) for n in table.model], ignore_index=True)
fc_long["y_hat"] = fc_long.y_hat.round(3)
out = C.RESULTS_DIR / "forecasts_test.csv.gz"
fc_long.to_csv(out, index=False, compression="gzip")
print(out.name, fc_long.shape, "|", fc_long.modelo.nunique(), "modelos"); fc_long.head()
""")


# =============================================================================
md(r"""
## 8. Discusión y análisis

**¿Qué modelo ganó y por cuánto superó al seasonal naive?** El mejor fue **Chronos-Bolt
zero-shot** (MASE 0.737, WAPE 5.9 %), 29 % mejor que el seasonal naive de 24 h (1.034) y 34 % mejor
que el de 168 h (1.114). Entre los modelos entrenados ganó **iTransformer** (0.803), seguido muy de
cerca por **LightGBM** (0.810) y nuestro **MLP** (0.819): la diferencia entre los tres (≤ 0.016) es del
orden de la desviación por semilla (iTransformer 0.812 ± 0.010; MLP 0.812 ± 0.007), así que en la
práctica empatan. En costo, el MLP entrena en 14 s y LightGBM en 105 s, contra 8.5 min de
iTransformer y 26–61 min de N-HiTS/TCN en CPU. En producción la mejora de ~22 % sobre el naive sí
justifica un modelo, pero no uno caro: LightGBM o el MLP están en la frontera de Pareto y son fáciles
de reentrenar; Chronos no se entrena pero exige ~205 M de parámetros e inferencia de 66 s por backtest.

**¿Se replica el hallazgo de DLinear?** Sí. Con H = 24, DLinear (0.866) y NLinear (0.868), con 16 k y
8 k parámetros, superan a PatchTST (0.916, 445 k parámetros y 46 min) y a MSTL; solo iTransformer, que
explota la correlación entre clientes, queda por delante. Con H = 168 la ventaja es todavía más
clara: DLinear 0.984, NLinear 0.994 y MLP 0.993 frente a Chronos 0.917 y seasonal naive 1.119.

**Estrategias multi-paso.** Con el mismo presupuesto (40 épocas) el **encoder–decoder** fue la mejor
(val 0.971; prueba 0.939), seguido de multi-salida (0.997 / 0.964) y autorregresiva (1.004 / 0.976).
La autorregresiva muestra la acumulación de errores: es la **mejor en h = 1** (MASE por paso 0.66 vs
0.74–0.77), porque su objetivo de entrenamiento es justamente el paso siguiente, pero a partir de
h ≈ 9–12 queda por encima de las otras dos (1.11 vs 1.02 en h = 12) al realimentar sus propios errores. El decoder que realimenta su predicción con teacher forcing al 50 % fue el peor (1.022): la
*exposure bias* le cuesta más de lo que aporta. El decoder no autorregresivo, que solo recibe el
calendario futuro, evita el problema y aprovecha las covariables. En todas las familias el error
crece con *h*, mezclado con la dificultad de cada hora del día (el máximo se da entre 14 y 19 h, donde
la carga es mayor y más variable, y baja de nuevo hacia h = 24). Los modelos que ven el último valor
(LightGBM, Chronos, iTransformer) arrancan muy bajo (MASE 0.48–0.56 en h = 1); DeepAR, que muestrea
trayectorias autorregresivas, casi duplica su error en el horizonte (0.78 → 1.58).

**Ventana L, normalización e historia.** Una L más larga ayudó a los lineales y al MLP (DLinear
0.958 → 0.883 → 0.862 con L = 48/168/336) pero no a la LSTM (1.014 con 168 vs 1.077 con 336: la
recurrencia olvida). La **normalización global** fue la peor en ambos modelos (DLinear 1.171, LSTM
1.752) porque los clientes grandes dominan la pérdida; z-score por serie y RevIN quedaron
prácticamente igual (0.883 vs 0.887). La historia importa mucho para la LSTM (1.264 → 1.088 → 1.014
con 3/12/30 meses) y para LightGBM (0.912 → 0.847 → 0.842), poco para DLinear (≈ 0.89) y nada para
PatchTST (0.93–0.96). Con 3 meses DLinear supera a LightGBM; desde 12 meses LightGBM supera a DLinear, la LSTM y
PatchTST en validación, y solo el MLP con 30 meses (0.818) lo supera; en prueba lo igualan el MLP e
iTransformer. Ningún modelo
de DL entrenado alcanzó a Chronos (0.755 en validación).

**¿Global o local?** El **global** ganó: MASE 0.858 contra 0.911 en los 10 clientes. Los que más se
benefician son clientes con series ruidosas o con cambios de régimen (MT_247, MT_268, MT_319,
MT_360: 11–14 % de mejora), que con su propia historia sobreajustan; los clientes muy regulares
(MT_046, MT_050, MT_279) quedan igual.

**Chronos zero-shot.** Quedó **por delante** de todos los modelos entrenados y además entrega
cuantiles bien calibrados (cobertura P10–P90 = 79.8 % ≈ 80 %, pinball 10.6 frente a 18.4 de DeepAR).
Lo usaríamos cuando no hay tiempo o datos para entrenar (clientes nuevos, arranque en frío, muchas
series heterogéneas) o como baseline fuerte antes de invertir en un modelo propio. Un modelo
entrenado conviene cuando hay covariables propias (clima, tarifas) o la latencia/costo de inferencia
importan.

**Navidad y Año Nuevo.** El 25-dic el consumo cae 36–54 % respecto a la segunda semana de diciembre.
En la semana del 22 al 28 de diciembre **todos** los modelos empeoran (MASE ×1.4 a ×2.2). Los que
tienen la bandera de feriado o usan el calendario (iTransformer 1.28, Chronos 1.34, N-HiTS 1.40,
LightGBM 1.41, LSTM 1.47) degradan mucho menos que los que no lo usan (DLinear/NLinear 1.82, MSTL 2.06,
seasonal naive 24 h 2.17). La covariable ayuda, pero con solo dos Navidades en el entrenamiento el
cambio de régimen sigue dominando el error.

**Reentrenamiento mensual.** Reentrenar el 1 de cada mes con ventana expandible casi no cambió el
error: LightGBM 0.810 → 0.805 y iTransformer 0.803 → 0.818 (dentro de la variación por semilla). Tres
meses no son suficientes para que la deriva de la serie haga necesario reentrenar.

**Sensibilidad a la semilla.** MLP: 0.8117 ± 0.0070; iTransformer: 0.8123 ± 0.0098 (3 semillas). Las
diferencias menores a ~0.01–0.02 de MASE entre modelos no deben interpretarse como significativas.

**Correlación entrenamiento–validación.** Una pérdida de entrenamiento más baja no implica siempre un
mejor MASE de validación (Spearman 0.65 entre ambas): el caso más claro es la
**LSTM autorregresiva**, que tiene la menor pérdida de entrenamiento de todas las iteraciones (0.163)
y un MASE de validación de 1.004, contra 0.204 y 0.819 del mejor MLP. Su pérdida se mide con teacher
forcing (un paso adelante con el valor real), que no refleja el error al realimentar sus propias
predicciones; lo mismo le pasa al seq2seq que realimenta y (0.194 → 1.022). Dentro de una misma
arquitectura sí hay relación, pero entre arquitecturas la pérdida de entrenamiento no predice el MASE.
""")


# =============================================================================
if __name__ == "__main__":
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = Path(__file__).resolve().parent / "lab9_series_tiempo.ipynb"
    nbf.write(nb, out)
    print(f"{out.name}: {len(CELLS)} celdas")
