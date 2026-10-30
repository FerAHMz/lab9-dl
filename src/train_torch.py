"""Entrenamiento y pronóstico de los modelos propios en PyTorch.

- Normalización: 'series' (z-score por cliente con estadísticos de entrenamiento),
  'global' (una sola media/desv. de todo el bloque de entrenamiento) o 'window'
  (RevIN sin parámetros: media/desv. de cada ventana de entrada).
- Entrenamiento con muestras aleatorias de ventanas de entrenamiento (stride 1);
  early stopping con los 92 orígenes diarios del bloque de VALIDACIÓN.
"""
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field

# Tope de memoria del allocator MPS (~40 % de la RAM recomendada) para no saturar la Mac.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.4")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config as C
from .data import SeriesScaler, build_panel, calendar_features
from .evaluation import Forecaster, daily_origins, mase_scale
from .torch_models import MODELS, count_params


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Panel:
    """Panel horario en memoria + índices de las particiones."""

    def __init__(self, panel=None):
        self.df = build_panel() if panel is None else panel
        self.index = self.df.index
        self.columns = self.df.columns
        self.Y = self.df.to_numpy("float32")                       # (T, N) kWh
        self.cal = calendar_features(self.index).to_numpy()       # (T, N_CAL)
        pos = lambda ts: int(self.index.get_loc(pd.Timestamp(ts)))
        self.t_train_end = pos(C.TRAIN_END)
        self.t_val, self.t_val_end = pos(C.VAL_START), pos(C.VAL_END)
        self.t_test, self.t_test_end = pos(C.TEST_START), pos(C.TEST_END)
        self.val_origins = daily_origins(self.index, C.VAL_START, C.VAL_END)
        self.test_origins = daily_origins(self.index, C.TEST_START, C.TEST_END)
        self.scale = mase_scale(self.Y, self.t_train_end)          # MASE (m=24)

    @property
    def N(self):
        return self.Y.shape[1]


class Normalizer:
    def __init__(self, mode, train_values):
        self.mode = mode
        if mode == "series":
            self.s = SeriesScaler().fit(train_values)
        elif mode == "global":
            self.mu, self.sd = float(train_values.mean()), float(train_values.std())

    def base(self, Y, cols):                     # Y: (..., n) para las series `cols`
        if self.mode == "series":
            return (Y - self.s.mu[cols]) / self.s.sd[cols]
        if self.mode == "global":
            return (Y - self.mu) / self.sd
        return Y

    def base_inv(self, Z, cols):
        if self.mode == "series":
            return Z * self.s.sd[cols] + self.s.mu[cols]
        if self.mode == "global":
            return Z * self.sd + self.mu
        return Z


@dataclass
class TrainConfig:
    model: str = "DLinear"
    L: int = C.L
    H: int = C.H
    months: int = 30                 # meses de historia de entrenamiento (hasta 2014-06-30)
    norm: str = "series"
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch: int = 512
    epochs: int = 30
    samples_per_epoch: int = 65536
    patience: int = 5
    loss: str = "l1"
    seed: int = C.SEED
    series: list = field(default=None)       # None = global (todas las series)
    extra: dict = field(default_factory=dict)

    def tag(self):
        s = "" if self.series is None else f"-local{self.series[0]}"
        s += "_fp" if self.extra.get("feed_prev") else ""
        s += f"_ep{self.epochs}" if self.extra.get("long") else ""
        return (f"{self.model}_L{self.L}_H{self.H}_m{self.months}_{self.norm}_h{self.hidden}"
                f"_l{self.layers}_d{self.dropout}_lr{self.lr}_b{self.batch}_s{self.seed}{s}")


class TorchForecaster(Forecaster):
    """Envuelve un modelo entrenado con la interfaz común de backtest()."""
    quantiles = None

    def __init__(self, name, net, cfg, panel, normalizer, device):
        self.name, self.net, self.cfg, self.P = name, net.eval(), cfg, panel
        self.norm, self.device = normalizer, device
        self.cols = np.arange(panel.N) if cfg.series is None else np.asarray(cfg.series)

    @torch.no_grad()
    def predict_batch(self, Y_in, t0s, cols):
        """Y_in: (B, L) kWh; t0s: (B,) orígenes; cols: (B,) cliente de cada fila."""
        L, H = self.cfg.L, self.cfg.H
        Z = self.norm.base(Y_in, cols[:, None])
        mu = sd = None
        if self.cfg.norm == "window":
            mu, sd = Z.mean(1, keepdims=True), Z.std(1, keepdims=True) + 1e-3
            Z = (Z - mu) / sd
        cal = self.P.cal
        Xc = np.stack([cal[t - L:t] for t in t0s])
        Xf = np.stack([cal[t:t + H] for t in t0s])
        X = np.concatenate([Z[..., None], Xc], axis=-1)
        out = self.net(torch.as_tensor(X, device=self.device, dtype=torch.float32),
                       torch.as_tensor(Xf, device=self.device, dtype=torch.float32))
        out = out.float().cpu().numpy()
        if self.cfg.norm == "window":
            out = out * sd + mu
        return self.norm.base_inv(out, cols[:, None]) if self.norm.mode == "series" \
            else self.norm.base_inv(out, None)

    def forecast(self, hist, t0):
        Y_in = hist[-self.cfg.L:, self.cols].T                          # (n, L)
        return self.predict_batch(Y_in, np.full(len(self.cols), t0), self.cols)


def _gather(Zb, cal, t0s, cols, L, H, norm_mode):
    """Construye un lote de ventanas de forma vectorizada (misma lógica que WindowDataset)."""
    ar_in = t0s[:, None] + np.arange(-L, 0)[None]
    ar_out = t0s[:, None] + np.arange(0, H)[None]
    y_in, y_out = Zb[ar_in, cols[:, None]], Zb[ar_out, cols[:, None]]
    mu = sd = None
    if norm_mode == "window":
        mu, sd = y_in.mean(1, keepdims=True), y_in.std(1, keepdims=True) + 1e-3
        y_in, y_out = (y_in - mu) / sd, (y_out - mu) / sd
    X = np.concatenate([y_in[..., None], cal[ar_in]], axis=-1)
    return X, cal[ar_out], y_out, (mu, sd)


def train_model(cfg: TrainConfig, panel: Panel, device=None, verbose=True,
                t_train_end=None, val_block=None):
    """Entrena con el bloque de entrenamiento (últimos `months` meses hasta 2014-06-30)
    y hace early stopping con el MASE de validación. Devuelve (forecaster, log).
    `t_train_end` y `val_block=(t_ini, t_fin)` permiten mover el corte (reentrenamiento
    mensual con ventana expandible)."""
    device = device or get_device()
    set_seed(cfg.seed)
    P, L, H = panel, cfg.L, cfg.H
    cols_all = np.arange(P.N) if cfg.series is None else np.asarray(cfg.series)
    t_end = P.t_train_end if t_train_end is None else t_train_end
    v_lo, v_hi = (P.t_val, P.t_val_end) if val_block is None else val_block
    t_start = max(0, t_end + 1 - cfg.months * 30 * 24 - L)  # +L de contexto inicial
    # El normalizador solo ve el bloque de entrenamiento (sin validación ni prueba).
    normalizer = Normalizer(cfg.norm, P.Y[t_start:t_end + 1])
    Zb = normalizer.base(P.Y, np.arange(P.N)).astype("float32")      # (T, N)
    # orígenes de entrenamiento: el objetivo completo debe caer en <= TRAIN_END
    tr_origins = np.arange(t_start + L, t_end - H + 2)

    net = MODELS[cfg.model](L, H, **_model_kwargs(cfg)).to(device)
    n_tot, n_tr = count_params(net)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    crit = nn.L1Loss() if cfg.loss == "l1" else nn.MSELoss()
    rng = np.random.default_rng(cfg.seed)
    teacher = cfg.model in ("LSTM-AR",) or (cfg.model == "LSTM-Seq2Seq"
                                            and cfg.extra.get("feed_prev"))

    fc = TorchForecaster(cfg.model, net, cfg, P, normalizer, device)
    # validación: 92 orígenes diarios del bloque jul-sep (todas las series del modelo)
    # (con H > 24 se descartan orígenes cuyo horizonte invadiría el bloque de prueba)
    val_origins = np.arange(v_lo, v_hi - H + 2, 24)
    v_t0 = np.repeat(val_origins, len(cols_all))
    v_cols = np.tile(cols_all, len(val_origins))
    Xv, Xfv, yv, v_stats = _gather(Zb, P.cal, v_t0, v_cols, L, H, cfg.norm)
    A_val = np.stack([P.Y[t:t + H, cols_all].T for t in val_origins], axis=1)  # (n,O,H)

    best, best_state, wait, log = np.inf, None, 0, []
    if device.type == "mps":
        torch.mps.empty_cache()
    peak_mem = 0
    t_total = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        net.train()
        tic, losses = time.perf_counter(), []
        n_batches = max(1, cfg.samples_per_epoch // cfg.batch)
        for _ in range(n_batches):
            t0s = rng.choice(tr_origins, cfg.batch)
            cols = rng.choice(cols_all, cfg.batch)
            X, Xf, y, _ = _gather(Zb, P.cal, t0s, cols, L, H, cfg.norm)
            X, Xf, y = (torch.as_tensor(a, device=device) for a in (X, Xf, y))
            pred = net(X, Xf, y) if teacher else net(X, Xf)
            loss = crit(pred, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if device.type == "mps":                 # memoria de tensores vivos en GPU
                peak_mem = max(peak_mem, torch.mps.current_allocated_memory())
        if device.type == "mps":                     # libera la caché del allocator
            torch.mps.empty_cache()
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
        ep_time = time.perf_counter() - tic

        # ---- validación (sin gradientes) ----
        net.eval()
        with torch.no_grad():
            pv = []
            for k in range(0, len(Xv), 512):          # lotes chicos: la LSTM en MPS reserva mucha memoria
                pv.append(net(torch.as_tensor(Xv[k:k + 512], device=device),
                              torch.as_tensor(Xfv[k:k + 512], device=device)).cpu())
            pv = torch.cat(pv)
            val_loss = crit(pv, torch.as_tensor(yv)).item()
        F_val = _to_kwh(pv.numpy(), v_cols, normalizer, v_stats).reshape(
            len(val_origins), len(cols_all), H).transpose(1, 0, 2)
        mae_c = np.abs(F_val - A_val).mean(axis=(1, 2))
        val_mae, val_mase = float(mae_c.mean()), float((mae_c / P.scale[cols_all]).mean())
        sched.step(val_mase)
        log.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=val_loss,
                        val_MAE=val_mae, val_MASE=val_mase, epoch_time=ep_time))
        if verbose:
            print(f"  ep {ep:02d} train {log[-1]['train_loss']:.4f} val {val_loss:.4f} "
                  f"MAE {val_mae:.2f} MASE {val_mase:.4f} ({ep_time:.1f}s)", flush=True)
        if val_mase < best - 1e-4:
            best, best_state, wait = val_mase, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    train_time = time.perf_counter() - t_total
    net.load_state_dict(best_state)
    info = dict(asdict(cfg), tag=cfg.tag(), extra=json.dumps(cfg.extra), params_total=n_tot, params_trainable=n_tr,
                best_val_MASE=best, best_epoch=int(np.argmin([r["val_MASE"] for r in log]) + 1),
                epochs_run=len(log), train_time=train_time,
                epoch_time_mean=float(np.mean([r["epoch_time"] for r in log])),
                peak_mem_MB=peak_mem / 2**20, device=str(device))
    info.pop("series")
    fc.net.eval()
    return fc, dict(info=info, epochs=log)


def _to_kwh(pred, cols, normalizer, stats):
    """Pasa predicciones normalizadas a kWh (deshace RevIN y luego la normalización base)."""
    mu, sd = stats
    out = pred if mu is None else pred * sd + mu
    return normalizer.base_inv(out, cols[:, None]) if normalizer.mode == "series" \
        else normalizer.base_inv(out, None)


def _model_kwargs(cfg):
    if cfg.model in ("DLinear", "NLinear"):
        return {}
    kw = dict(hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout)
    if cfg.model == "LSTM-Seq2Seq":
        kw.update(feed_prev=cfg.extra.get("feed_prev", False),
                  tf_ratio=cfg.extra.get("tf_ratio", 0.5))
    if cfg.model == "MLP":
        kw["hidden"] = cfg.extra.get("mlp_hidden", 512)
    return kw
