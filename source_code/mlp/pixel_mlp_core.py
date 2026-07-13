from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def _activation_module(name: str) -> nn.Module:
    n = (name or "relu").lower().strip()
    if n == "tanh":
        return nn.Tanh()
    return nn.ReLU()


def build_torch_mlp(in_dim: int, hidden_layer_sizes: Sequence[int], activation: str) -> nn.Module:
    layers: List[nn.Module] = []
    prev = int(in_dim)
    for h in hidden_layer_sizes:
        hi = int(h)
        layers.append(nn.Linear(prev, hi))
        layers.append(_activation_module(activation))
        prev = hi
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_torch_pixel_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_layer_sizes: Tuple[int, ...],
    activation: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tuple[nn.Module, StandardScaler]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X).astype(np.float32, copy=False)
    yv = y.astype(np.float32, copy=False).reshape(-1, 1)

    n = Xs.shape[0]
    in_dim = int(Xs.shape[1])
    net = build_torch_mlp(in_dim, hidden_layer_sizes, activation).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xs),
        torch.from_numpy(yv),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=min(int(batch_size), max(n, 1)),
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    net.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

    net.eval()
    return net, scaler


def _rebuild_net_from_arch(arch: Dict[str, Any]) -> nn.Module:
    return build_torch_mlp(
        int(arch["in_dim"]),
        tuple(int(x) for x in arch["hidden_layer_sizes"]),
        str(arch.get("activation", "relu")),
    )


def predict_from_bundle(
    bundle: Dict[str, Any],
    feats: np.ndarray,
    *,
    chunk: int = 200000,
    device: torch.device,
) -> np.ndarray:
    if bundle.get("backend") == "torch":
        scaler: StandardScaler = bundle["scaler"]
        arch = bundle["torch_arch"]
        z = scaler.transform(feats).astype(np.float32, copy=False)
        net = _rebuild_net_from_arch(arch).to(device)
        net.load_state_dict(bundle["torch_state_dict"])
        net.eval()
        out = np.zeros((z.shape[0],), dtype=np.float32)
        n = z.shape[0]
        i = 0
        with torch.inference_mode():
            while i < n:
                j = min(n, i + int(chunk))
                t = torch.from_numpy(z[i:j]).to(device, non_blocking=True)
                out[i:j] = net(t).squeeze(-1).detach().float().cpu().numpy()
                i = j
        return out

    model = bundle.get("model")
    if model is None:
        raise ValueError("Bundle has no sklearn model and backend is not torch.")
    o = np.zeros((feats.shape[0],), dtype=np.float32)
    n = feats.shape[0]
    i = 0
    while i < n:
        j = min(n, i + int(chunk))
        o[i:j] = model.predict(feats[i:j]).astype(np.float32, copy=False)
        i = j
    return o


def pick_torch_device(pref: str, gpu_id: int = 0) -> torch.device:
    p = (pref or "cpu").strip().lower()
    if p in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")
