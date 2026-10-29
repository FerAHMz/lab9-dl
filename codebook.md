# Codebook — ElectricityLoadDiagrams20112014

**Fuente:** UCI Machine Learning Repository, dataset 321
(<https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>).
Consumo eléctrico de 370 clientes de Portugal, 2011-01-01 a 2014-12-31, cada 15 min.

## Archivo crudo `data/raw/LD2011_2014.txt` (no versionado; lo descarga `src/data.py`)

| Columna | Tipo | Descripción | Rango |
|---|---|---|---|
| (índice) | datetime | Etiqueta del registro de 15 min (hora de Portugal) | 2011-01-01 00:15 → 2015-01-01 00:00 |
| `MT_001` … `MT_370` | float (coma decimal) | Potencia promedio del cliente en el cuarto de hora, en **kW** | 0 → ~70,000 |

140,256 filas × 370 clientes. Antes del alta de un cliente los valores son 0.

## Datos procesados `data/processed/` (no versionados, se regeneran con `src/01_prepare_data.py`)

| Archivo | Contenido |
|---|---|
| `hourly_all.parquet` | Consumo **horario en kWh** 2012-01-01 → 2014-12-31 (26,304 h) de los 314 clientes con datos completos |
| `hourly_panel.parquet` | Subconjunto de 100 clientes (semilla 42) usado por todos los modelos |

## Variables del modelo

| Variable | Tipo | Descripción | Valores |
|---|---|---|---|
| `y` | float | Consumo por hora por cliente, kWh = Σ kW · 0.25 h de los 4 cuartos HH:00–HH:45 | 0 → ~17,000 |
| `hour_sin`, `hour_cos` | float | Hora del día en seno/coseno (período 24) | [-1, 1] |
| `dow_sin`, `dow_cos` | float | Día de la semana (lunes = 0) en seno/coseno (período 7) | [-1, 1] |
| `month_sin`, `month_cos` | float | Mes en seno/coseno (período 12) | [-1, 1] |
| `holiday` | binaria | Feriado nacional de Portugal (librería `holidays`, país `PT`) | {0, 1} |
| `weekend` | binaria | Sábado o domingo | {0, 1} |
| `unique_id` | str | Identificador del cliente (formato largo de Nixtla) | `MT_xxx` |
| `ds` | datetime | Inicio de la hora | 2012-01-01 00:00 → 2014-12-31 23:00 |

## Transformaciones aplicadas

1. Agregación 15 min → hora (kW → kWh).
2. Cambios de horario: hora 01:00 del último domingo de marzo (en cero) imputada con el
   promedio de 00:00 y 02:00; hora 01:00 del último domingo de octubre (doble) dividida entre 2.
3. Filtro: clientes sin rachas de ceros > 24 h en 2012–2014 (314 de 370).
4. Muestra fija de 100 clientes (semilla 42).
5. Normalización z-score por cliente con media/desv. del bloque de entrenamiento
   (2012-01-01 → 2014-06-30).

## Particiones

| Bloque | Periodo | Uso |
|---|---|---|
| Entrenamiento | 2012-01-01 → 2014-06-30 | Ajuste de pesos |
| Validación | 2014-07-01 → 2014-09-30 | Selección de hiperparámetros y early stopping |
| Prueba | 2014-10-01 → 2014-12-31 | Backtest con 92 orígenes diarios (H = 24) |
