"""
Per-channel forecasting detector.

A model predicts the next telemetry value from a window of recent values; the
prediction error is the anomaly signal handed to the dynamic thresholder. Two
backends:

  - "fast": ridge regression over lagged windows. Pure CPU, no deep-learning
    deps, runs in seconds per channel. Good for development and CI.
  - "lstm": stacked LSTM forecaster (PyTorch), matching the architecture family
    of the JPL baseline. This is the backend for the reported benchmark. Auto-
    used when torch is installed and --backend lstm is passed.

Both produce the same downstream error signal, so the multi-agent layer and
evaluation are backend-agnostic.
"""

import numpy as np

WINDOW = 100      # look-back length
HORIZON = 1       # predict next step


def _windows(x, w):
    n = len(x) - w
    if n <= 0:
        return np.empty((0, w)), np.empty((0,))
    X = np.stack([x[i:i + w] for i in range(n)])
    y = x[w:w + n]
    return X, y


def prediction_errors(signal, backend="fast", epochs=10):
    """signal: (T,1) array. Returns per-timestep absolute prediction error (T,)."""
    x = signal[:, 0].astype(float)
    x = (x - x.mean()) / (x.std() + 1e-9)
    if backend == "lstm":
        preds = _lstm_forecast(x, epochs=epochs)
    else:
        preds = _fast_forecast(x)
    err = np.zeros_like(x)
    err[WINDOW:WINDOW + len(preds)] = np.abs(x[WINDOW:WINDOW + len(preds)] - preds)
    return err


def _fast_forecast(x):
    from sklearn.linear_model import Ridge
    X, y = _windows(x, WINDOW)
    if len(X) == 0:
        return np.array([])
    split = max(1, int(0.3 * len(X)))            # train on early, predict all
    model = Ridge(alpha=1.0)
    model.fit(X[:split], y[:split])
    return model.predict(X)


def _lstm_forecast(x, epochs=10):
    import torch
    import torch.nn as nn

    X, y = _windows(x, WINDOW)
    if len(X) == 0:
        return np.array([])
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    split = max(1, int(0.3 * len(X)))

    class LSTMForecaster(nn.Module):
        def __init__(self, hidden=64, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=0.3)
            self.head = nn.Linear(hidden, 1)

        def forward(self, z):
            out, _ = self.lstm(z)
            return self.head(out[:, -1, :])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = LSTMForecaster().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xtr, ytr = Xt[:split].to(dev), yt[:split].to(dev)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        preds = net(Xt.to(dev)).cpu().numpy().ravel()
    return preds
