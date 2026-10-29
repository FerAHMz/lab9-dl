"""Rutas, fechas y constantes compartidas por todo el pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figs"

RAW_URL = ("https://archive.ics.uci.edu/static/public/321/"
           "electricityloaddiagrams20112014.zip")
RAW_TXT = RAW_DIR / "LD2011_2014.txt"
HOURLY_ALL = PROC_DIR / "hourly_all.parquet"      # todos los clientes válidos
HOURLY_PANEL = PROC_DIR / "hourly_panel.parquet"  # subconjunto de N_CLIENTS

SEED = 42
N_CLIENTS = 100

# Particiones (inclusive). El índice horario marca el INICIO de la hora.
START = "2012-01-01 00:00"
TRAIN_END = "2014-06-30 23:00"
VAL_START, VAL_END = "2014-07-01 00:00", "2014-09-30 23:00"
TEST_START, TEST_END = "2014-10-01 00:00", "2014-12-31 23:00"
XMAS_START, XMAS_END = "2014-12-22 00:00", "2014-12-28 23:00"

L = 168          # ventana de entrada por defecto
H = 24           # horizonte day-ahead
SEASON = 24      # estacionalidad para el escalamiento de MASE
QUANTILES = [0.1, 0.5, 0.9]
