"""
Per-channel forecasting detector.

A model predicts the next telemetry value (column 0) from a window of recent
multivariate inputs (telemetry + command-input context, columns 0..F-1); the
prediction error on the test signal is the anomaly signal handed to the
dynamic thresholder. Two backends:

  - "fast": ridge regression over flattened multivariate windows. Pure CPU,
    no deep-learning deps, runs in seconds per channel. Good for development
    and CI.
  - "lstm": stacked LSTM forecaster (PyTorch), matching the architecture
    family of the JPL baseline. Mini-batched over the full NASA train signal;
    this is the backend for the reported benchmark.

Both produce the same downstream error signal, so the multi-agent layer and
evaluation are backend-agnostic.

If a dedicated NASA train signal is provided (real-data mode), the forecaster
is fit on it and scored on the disjoint test signal. If no train signal is
given (synthetic smoke-test mode), the model falls back to self-split: fit on
the first 30% of test windows, score on all.
"""

import numpy as np

WINDOW = 100      # look-back length
HORIZON = 1       # predict next step


def _windows_multi(x_2d, w):
    """x_2d: (T, F). Returns X: (n, w, F) windows and y: (n,) next-step
    telemetry (column 0)."""
    n = len(x_2d) - w
    if n <= 0:
        return np.empty((0, w, x_2d.shape[1])), np.empty((0,))
    X = np.stack([x_2d[i:i + w] for i in range(n)])
    y = x_2d[w:w + n, 0]
    return X, y


def _normalize(test_2d, train_2d=None):
    """Z-score only column 0 (telemetry target) using TEST signal stats; leave
    columns 1+ (one-hot command inputs) raw. We use test stats rather than
    train stats because ~30% of channels have a near-constant train col 0
    (std<0.01) which would otherwise blow up the test signal under z-scoring.
    Test col 0 always has variance because it contains the anomaly windows."""
    test_2d = np.array(test_2d, dtype=float, copy=True)
    mu = test_2d[:, 0].mean()
    sigma = test_2d[:, 0].std() + 1e-9
    test_2d[:, 0] = (test_2d[:, 0] - mu) / sigma

    if train_2d is not None and len(train_2d) > 0:
        train_2d = np.array(train_2d, dtype=float, copy=True)
        # Defensive: align feature dim if mismatched (NASA channels match by design)
        if train_2d.shape[1] != test_2d.shape[1]:
            min_F = min(train_2d.shape[1], test_2d.shape[1])
            train_2d = train_2d[:, :min_F]
            test_2d = test_2d[:, :min_F]
        train_2d[:, 0] = (train_2d[:, 0] - mu) / sigma
        return test_2d, train_2d
    return test_2d, None


def prediction_errors(signal, backend="fast", epochs=10, train_signal=None):
    """signal: (T, F) test array (F>=1). train_signal: optional (T_train, F).
    Returns per-timestep absolute prediction error (T,). When train_signal is
    None, falls back to self-split on the test signal (legacy / synthetic)."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    test_n, train_n = _normalize(signal, train_signal)

    if backend == "lstm":
        preds = _lstm_forecast(test_n, train_n, epochs=epochs)
    else:
        preds = _fast_forecast(test_n, train_n)

    err = np.zeros(len(test_n), dtype=float)
    target = test_n[:, 0]
    err[WINDOW:WINDOW + len(preds)] = np.abs(target[WINDOW:WINDOW + len(preds)] - preds)
    return err


def _fast_forecast(test_n, train_n=None):
    from sklearn.linear_model import Ridge
    X_test, _ = _windows_multi(test_n, WINDOW)
    if len(X_test) == 0:
        return np.array([])

    if train_n is not None and len(train_n) > WINDOW:
        X_train, y_train = _windows_multi(train_n, WINDOW)
    else:
        # synthetic / no-train fallback: self-split on test
        X_all, y_all = _windows_multi(test_n, WINDOW)
        split = max(1, int(0.3 * len(X_all)))
        X_train, y_train = X_all[:split], y_all[:split]

    if len(X_train) == 0:
        return np.array([])

    # ridge over flattened multivariate windows
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, y_train)
    return model.predict(Xte)


def _lstm_forecast(test_n, train_n=None, epochs=10, batch_size=256, hidden=80):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_test, _ = _windows_multi(test_n, WINDOW)
    if len(X_test) == 0:
        return np.array([])
    F = X_test.shape[2]

    if train_n is not None and len(train_n) > WINDOW:
        X_train, y_train = _windows_multi(train_n, WINDOW)
    else:
        X_all, y_all = _windows_multi(test_n, WINDOW)
        split = max(1, int(0.3 * len(X_all)))
        X_train, y_train = X_all[:split], y_all[:split]

    if len(X_train) == 0:
        return np.array([])

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    class LSTMForecaster(nn.Module):
        def __init__(self, in_features, h, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(in_features, h, layers,
                                batch_first=True, dropout=0.3)
            self.head = nn.Linear(h, 1)

        def forward(self, z):
            out, _ = self.lstm(z)
            return self.head(out[:, -1, :])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = LSTMForecaster(F, hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=batch_size, shuffle=True)
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch_size):
            batch = Xte[i:i + batch_size].to(dev)
            preds.append(net(batch).cpu().numpy().ravel())
    return np.concatenate(preds) if preds else np.array([])