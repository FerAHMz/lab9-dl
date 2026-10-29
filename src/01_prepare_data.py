"""Etapa 1: crudo (15 min) -> horario limpio -> panel de 100 clientes (data/processed/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                      # noqa: E402
from src.data import build_hourly, build_panel   # noqa: E402

if __name__ == "__main__":
    allc = build_hourly(force=True)
    panel = build_panel(force=True)
    print(f"clientes completos 2012-2014: {allc.shape[1]}  |  horas por serie: {allc.shape[0]}")
    print(f"panel: {panel.shape[1]} clientes x {panel.shape[0]} h = {panel.size:,} observaciones")
    print(f"guardado en {C.HOURLY_PANEL}")
