# Laboratorio 9 — Series de tiempo y forecasting (CC3092)

Pronóstico *day-ahead* (H = 24 h) del consumo eléctrico horario de 100 clientes de Portugal
(ElectricityLoadDiagrams20112014, UCI) con baselines estadísticos, LightGBM, modelos propios en
PyTorch (MLP, DLinear, NLinear, LSTM con tres estrategias multi-paso y seq2seq), modelos de
NeuralForecast (N-HiTS, TCN, PatchTST, iTransformer, DeepAR) y Chronos-Bolt zero-shot. Todos se
evalúan con la misma función `backtest()` sobre 92 orígenes diarios de oct–dic 2014.

**Integrantes:** Fernando Andree Hernández Martínez (23645) · Fernando José Rueda Rodas (23748)

## Estructura

```
lab9-dl/
├── src/
│   ├── config.py               # rutas, fechas de las particiones, L, H, semilla
│   ├── data.py                 # crudo -> horario -> horario de verano -> panel -> calendario -> WindowDataset
│   ├── evaluation.py           # backtest(), métricas (MAE, RMSE, WAPE, MAPE, MASE, pinball)
│   ├── torch_models.py         # MLP, DLinear, NLinear, LSTM (AR / MIMO / seq2seq)
│   ├── train_torch.py          # entrenamiento con early stopping en validación
│   ├── library_models.py       # adaptadores StatsForecast, LightGBM, NeuralForecast, Chronos
│   ├── utils.py                # persistencia de resultados y bitácora de iteraciones
│   ├── 01_prepare_data.py      # etapa 1: panel horario de 100 clientes
│   ├── 02_baselines.py         # etapa 2: Naive, SeasonalNaive 24/168, MSTL
│   ├── 03_lightgbm.py          # etapa 3: LightGBM global (historia 3/12/30 meses)
│   ├── 04_torch_models.py      # etapa 4: iteraciones de hiperparámetros + evaluación (--eval)
│   ├── 05_neuralforecast.py    # etapa 5: modelos de NeuralForecast (--history para PatchTST)
│   ├── 06_chronos.py           # etapa 6: Chronos-Bolt zero-shot
│   └── 07_extras.py            # etapa 7: --h168 --seeds --local --retrain
├── notebooks/
│   ├── lab9_series_tiempo.ipynb  # notebook completo y ejecutado
│   └── build_notebook.py         # genera el notebook a partir de sus celdas
├── results/
│   ├── runs/                   # pronósticos de cada backtest (.npz) + metadatos (.json)
│   ├── iterations.csv          # configuración y métricas de cada iteración
│   ├── epoch_logs.csv          # pérdida de entrenamiento/validación y MASE por epoch
│   ├── comparison_table.csv    # tabla comparativa final
│   └── forecasts_test.csv.gz   # pronósticos de prueba (unique_id, ds, modelo, y_hat)
├── report/                     # informe PDF y figuras
├── data/raw/                   # LD2011_2014.txt (se descarga; no se versiona)
├── data/processed/             # parquet horarios (se regeneran; no se versionan)
├── codebook.md
└── requirements.txt
```

## Entorno

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución (desde la raíz del repo, en orden)

```bash
python src/01_prepare_data.py                 # descarga el crudo y arma el panel
python src/02_baselines.py                    # ~40 min (MSTL se reajusta por origen)
python src/03_lightgbm.py                     # ~6 min
python src/04_torch_models.py                 # iteraciones (cada una en su propio proceso)
python src/04_torch_models.py --eval          # mejor config. de cada modelo -> prueba
python src/05_neuralforecast.py               # N-HiTS, TCN, PatchTST, iTransformer, DeepAR
python src/05_neuralforecast.py --history     # PatchTST con 3/12/30 meses (validación)
python src/06_chronos.py                      # Chronos-Bolt zero-shot
python src/07_extras.py --h168 --seeds --local --retrain
python notebooks/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/lab9_series_tiempo.ipynb
```

Los modelos se ejecutan en secuencia (uno a la vez) para no saturar la RAM. Hardware usado:
Apple M4 Pro, 24 GB (PyTorch en MPS; NeuralForecast, LightGBM y StatsForecast en CPU).

## Contenido del notebook

1. Datos: carga, agregación horaria, cambios de horario, filtro 2012–2014.
2. Exploración: perfiles por hora y día, feriados, Navidad/Año Nuevo, escalas, normalización,
   covariables de calendario, `WindowDataset` y conteo de ventanas, auditoría de fugas.
3. Investigación: LSTM/GRU, Conv1d/TCN, Transformer, Nixtla, Chronos, local vs. global,
   teacher forcing, RevIN, pinball loss, MASE vs. WAPE.
4. Protocolo: `backtest()`, métricas y comportamiento del MAPE.
5. Modelos: implementaciones propias, estrategias multi-paso, iteraciones, curvas de pérdida,
   local vs. global.
6. Comparación: tabla final, frontera de Pareto, MAE por paso, MASE vs. historia, resultados
   probabilísticos, Navidad, reentrenamiento mensual, semillas y archivo de pronósticos.
7. Discusión y conclusiones.
