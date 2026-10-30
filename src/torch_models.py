"""Implementaciones propias en PyTorch: MLP, DLinear, NLinear, LSTM (3 estrategias) y seq2seq.

Convención de entradas para todos los modelos:
  X   (B, L, 1 + N_CAL): canal 0 = consumo normalizado, resto = calendario del pasado
  Xf  (B, H, N_CAL)    : calendario de las H horas a pronosticar (conocido a futuro)
Salida: (B, H) en el mismo espacio normalizado que el canal 0.
"""
import torch
import torch.nn as nn

from .data import N_CAL


class MLP(nn.Module):
    """MLP multi-salida: [ventana L de consumo, calendario futuro aplanado] -> 24 pasos."""
    strategy, uses_cov = "multi-salida", True

    def __init__(self, L, H, hidden=512, layers=2, dropout=0.1):
        super().__init__()
        dims, mods = L + H * N_CAL, []
        for _ in range(layers):
            mods += [nn.Linear(dims, hidden), nn.ReLU(), nn.Dropout(dropout)]
            dims = hidden
        self.net = nn.Sequential(*mods, nn.Linear(dims, H))

    def forward(self, X, Xf):
        return self.net(torch.cat([X[..., 0], Xf.flatten(1)], dim=1))


class MovingAvg(nn.Module):
    """Media móvil con relleno de bordes (descomposición tendencia/estacional de DLinear)."""

    def __init__(self, k=25):
        super().__init__()
        self.k = k
        self.pool = nn.AvgPool1d(k, stride=1)

    def forward(self, x):                                   # (B, L)
        pad = (self.k - 1) // 2
        x = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, self.k - 1 - pad)], 1)
        return self.pool(x.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    """DLinear (Zeng et al., 2023): y = W_t * tendencia + W_s * resto. Sin covariables."""
    strategy, uses_cov = "multi-salida", False

    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.decomp = MovingAvg(kernel)
        self.lin_trend, self.lin_season = nn.Linear(L, H), nn.Linear(L, H)

    def forward(self, X, Xf):
        x = X[..., 0]
        trend = self.decomp(x)
        return self.lin_trend(trend) + self.lin_season(x - trend)


class NLinear(nn.Module):
    """NLinear: resta el último valor, capa lineal L->H, y lo vuelve a sumar."""
    strategy, uses_cov = "multi-salida", False

    def __init__(self, L, H):
        super().__init__()
        self.lin = nn.Linear(L, H)

    def forward(self, X, Xf):
        x = X[..., 0]
        last = x[:, -1:]
        return self.lin(x - last) + last


class LSTMMimo(nn.Module):
    """LSTM multi-salida: el último estado oculto + calendario futuro -> capa densa de 24."""
    strategy, uses_cov = "multi-salida", True

    def __init__(self, L, H, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(1 + N_CAL, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + H * N_CAL, hidden), nn.ReLU(),
                                  nn.Linear(hidden, H))

    def forward(self, X, Xf):
        _, (h_n, _) = self.rnn(X)                           # h_n: (layers, B, hidden)
        return self.head(torch.cat([h_n[-1], Xf.flatten(1)], dim=1))


class LSTMAutoregressive(nn.Module):
    """LSTM autorregresiva: entrada en t = [y_t, calendario_{t+1}] -> y_{t+1}.

    Entrenamiento con teacher forcing (se alimenta el valor real anterior en todas las
    posiciones); en inferencia se realimenta la propia predicción durante 24 pasos.
    """
    strategy, uses_cov = "autorregresiva", True

    def __init__(self, L, H, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.H = H
        self.rnn = nn.LSTM(1 + N_CAL, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, 1)

    def _inputs(self, y, cal_next):
        return torch.cat([y.unsqueeze(-1), cal_next], dim=-1)

    def forward(self, X, Xf, y_true=None):
        y_hist, cal_hist = X[..., 0], X[..., 1:]
        cal_next = torch.cat([cal_hist[:, 1:], Xf[:, :1]], dim=1)   # calendario de t+1
        if y_true is not None:                                      # teacher forcing
            y_all = torch.cat([y_hist, y_true[:, :-1]], dim=1)
            cal_all = torch.cat([cal_next, Xf[:, 1:]], dim=1)
            o, _ = self.rnn(self._inputs(y_all, cal_all))
            return self.out(o).squeeze(-1)[:, -self.H:]           # pérdida en los H pasos
        o, state = self.rnn(self._inputs(y_hist, cal_next))
        y_prev, preds = self.out(o[:, -1]).squeeze(-1), []
        preds.append(y_prev)
        for h in range(1, self.H):                                  # realimenta su predicción
            inp = self._inputs(y_prev.unsqueeze(1), Xf[:, h:h + 1])
            o, state = self.rnn(inp, state)
            y_prev = self.out(o[:, -1]).squeeze(-1)
            preds.append(y_prev)
        return torch.stack(preds, dim=1)


class LSTMSeq2Seq(nn.Module):
    """Encoder-decoder: el encoder LSTM resume la ventana; el decoder LSTM recibe en cada
    paso futuro el calendario conocido (hora, día, mes, feriado) y, opcionalmente, el
    valor anterior (teacher forcing con probabilidad `tf_ratio` durante entrenamiento)."""
    strategy, uses_cov = "encoder-decoder", True

    def __init__(self, L, H, hidden=128, layers=2, dropout=0.1, feed_prev=False,
                 tf_ratio=0.5):
        super().__init__()
        self.H, self.feed_prev, self.tf_ratio = H, feed_prev, tf_ratio
        self.enc = nn.LSTM(1 + N_CAL, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dec = nn.LSTM(N_CAL + int(feed_prev), hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, 1)

    def forward(self, X, Xf, y_true=None):
        _, state = self.enc(X)
        if not self.feed_prev:                         # decoder no autorregresivo
            o, _ = self.dec(Xf, state)
            return self.out(o).squeeze(-1)
        y_prev, preds = X[:, -1, 0], []
        for h in range(self.H):
            inp = torch.cat([y_prev[:, None, None], Xf[:, h:h + 1]], dim=-1)
            o, state = self.dec(inp, state)
            y_hat = self.out(o[:, -1]).squeeze(-1)
            preds.append(y_hat)
            use_tf = (y_true is not None and self.training
                      and torch.rand(()) < self.tf_ratio)
            y_prev = y_true[:, h] if use_tf else y_hat
        return torch.stack(preds, dim=1)


MODELS = {"MLP": MLP, "DLinear": DLinear, "NLinear": NLinear, "LSTM-MIMO": LSTMMimo,
          "LSTM-AR": LSTMAutoregressive, "LSTM-Seq2Seq": LSTMSeq2Seq}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
