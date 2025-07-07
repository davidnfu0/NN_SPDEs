"""heat_1d_solver.py  — clean, unsupervised version (rev‑USV)
============================================================
Fits the mild solution of the 1‑D stochastic heat equation in the exact set‑up
of Neufeld & Schmocker (Fig. 1, *m* = 1) – using the unsupervised variational
loss (drift + noise) as in the original paper.

Key facts
---------
* **Chaos truncation**  (I,J,K) = (1, 5, 1)
* **Time grid**          T = 0.25,  *n* = 100 internal time steps
* **Training grid**      M₂ = 20  (time),  M₃ = 1 000 (space)
* **Network**           2 hidden layers, 75 neurons each, ρ = tanh
* **Optimiser**         Adam (lr = 2⋅10⁻³, 5 000 epochs), batch size 40
* **Brownian noise**    single path ω₀, linear interpolation
Output is a 3‑D plot: colored surface = learned X, black wireframe = true X

"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from AlphaSet import AlphaSet
from NeuralNetwork import NeuralNetwork

# ‑‑‑ constants -------------------------------------------------------------
SIGMA = 6.0                 # width of Gaussian initial profile
BATCH_SIZE = 40


if not hasattr(torch, "version"):
    class _V:  # type: ignore
        hip = False
    torch.version = _V()  # type: ignore


def build_alpha_set(cache: str = "files/alpha_I1J5K1.pkl") -> AlphaSet:
    I, J, K, n, T, N_normals = 1, 5, 1, 100, 0.25, 5
    p = Path(cache)
    if p.exists():
        return pickle.loads(p.read_bytes())
    aset = AlphaSet(I, J, K, n, T)
    aset.calculate_alphas_fast()
    rng = np.random.default_rng(42)
    aset.add_normals(rng.standard_normal((N_normals, I, J)).astype(np.float32))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(aset))
    return aset


def _interp1d(t: torch.Tensor, t_grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    dt = t_grid[1] - t_grid[0]
    idx_float = (t / dt).clamp(0, t_grid.numel() - 1 - 1e-6)
    idx_lo = idx_float.floor().long()
    idx_hi = torch.clamp(idx_lo + 1, max=t_grid.numel() - 1)

    w_hi = (t - t_grid[idx_lo]) / dt
    return (1 - w_hi) * values[idx_lo] + w_hi * values[idx_hi]


def spde_loss(model, sd, ic, aset):
    device = next(model.parameters()).device
    t, u = sd[:, 0], sd[:, 1]
    sd = sd.to(device).requires_grad_(True)
    pred = model(sd)

    grad = torch.autograd.grad(pred, sd, grad_outputs=torch.ones_like(pred),
                               retain_graph=True, create_graph=True)[0]
    du = grad[:, 1]

    grad2 = torch.autograd.grad(du, sd, grad_outputs=torch.ones_like(du),
                                retain_graph=True, create_graph=True)[0]
    d2u = grad2[:, 1]

    t_vec = torch.tensor(aset.t, dtype=torch.float32, device=device)
    dt = t_vec[1] - t_vec[0]
    drift = dt * d2u
    X0 = 10.0 * torch.exp(-u ** 2 / (2 * SIGMA ** 2))

    W_path = torch.tensor(aset.brownian_paths[model.omega][0], dtype=torch.float32, device=device)
    W_interp = _interp1d(t, t_vec, W_path)
    if W_interp.ndim < u.ndim:
        W_interp = W_interp.unsqueeze(-1).expand_as(u)

    target = X0 + drift + W_interp
    return torch.mean((pred - target) ** 2)


def main():
    aset = build_alpha_set()
    t_train = torch.linspace(0.0, aset.T, 20)
    u_train = torch.linspace(-2.0, 2.0, 1000)
    Tm, Um = torch.meshgrid(t_train, u_train, indexing="ij")
    train_pts = torch.stack([Tm.flatten(), Um.flatten()], 1)

    net = NeuralNetwork(
        space_dim=2,
        alpha_set=aset,
        n_layers=2,
        wide=75,
        activation=torch.nn.Tanh,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    net.train()
    best_loss = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    train_pts = train_pts.to(device)

    for epoch in range(5000):
        perm = torch.randperm(train_pts.size(0))
        for i in range(0, train_pts.size(0), BATCH_SIZE):
            batch = train_pts[perm[i:i+BATCH_SIZE]]
            optimizer.zero_grad()
            loss = spde_loss(net, batch, None, aset)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                net.save_weights("heat1d_usv_best")
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Best Loss: {best_loss:.4f}")

    net.load_weights("heat1d_usv_best")

    t_grid = torch.linspace(0.0, aset.T, aset.n)
    u_grid = torch.linspace(-2.0, 2.0, 200)
    Tg, Ug = torch.meshgrid(t_grid, u_grid, indexing="ij")
    pts = torch.stack([Tg.flatten(), Ug.flatten()], 1).to(device)

    with torch.no_grad():
        X_pred = net(pts).reshape_as(Tg).cpu()

    # Reconstruct true solution
    sigma2 = 2 * Tg + SIGMA ** 2
    det_part = 10.0 * (SIGMA ** 2 / sigma2).sqrt() * torch.exp(-(Ug ** 2) / (2 * sigma2))
    W = torch.tensor(aset.brownian_paths[0][0], dtype=torch.float32)
    t_vec = torch.tensor(aset.t, dtype=torch.float32)
    W_interp = _interp1d(Tg, t_vec, W)
    if W_interp.ndim < Ug.ndim:
        W_interp = W_interp.unsqueeze(-1).expand_as(Ug)
    X_true = det_part + W_interp

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(Ug, Tg, X_true, color="k", linewidth=0.6, alpha=0.9)
    surf = ax.plot_surface(Ug, Tg, X_pred, cmap=cm.plasma, edgecolor="none", alpha=0.85)

    ax.view_init(elev=25, azim=120)
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$X_t(\\omega)(u_1)$")
    ax.set_title("Prediction (color) vs True (wireframe)")
    fig.colorbar(surf, shrink=0.55, aspect=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
