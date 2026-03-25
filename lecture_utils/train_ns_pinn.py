"""
train_ns_pinn.py
================
Overnight training script for the NS PINN on the cylinder wake dataset.

INSTRUCTOR USE ONLY — run this on Mjolnir before the lecture to generate
the pre-trained model file used in Lecture 2.

Usage
-----
    conda activate pinn-heat
    cd lectures/
    python lecture_utils/train_ns_pinn.py

Output
------
    data/ns_pinn_pretrained.pt   — saved model state dict
    data/ns_pinn_loss_history.npz — training loss history for plotting

Estimated runtime
-----------------
    NVIDIA RTX A2000 (Mjolnir): ~2–3 hours
    CPU only: not recommended (12+ hours)

Dataset
-------
    data/cylinder_nektar_wake.mat
    Download from:
    https://github.com/maziarraissi/PINNs/tree/master/main/Data

Reference
---------
    Raissi, M., Yazdani, A., & Karniadakis, G.E. (2020).
    Hidden fluid mechanics. Science, 367(6481), 1026-1030.
    https://doi.org/10.1126/science.aaw4741
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import scipy.io

# ── Add parent directory so lecture_utils is importable ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lecture_utils.pinn import NS_PINN
from lecture_utils.config import SEED

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device  : {device}")
print(f"PyTorch : {torch.__version__}")

# ── Load dataset ──────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data',
                           'cylinder_nektar_wake.mat')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                           'ns_pinn_pretrained.pt')
LOSS_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data',
                           'ns_pinn_loss_history.npz')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Download from: https://github.com/maziarraissi/PINNs/tree/master/main/Data"
    )

print(f"\nLoading dataset: {DATA_PATH}")
data   = scipy.io.loadmat(DATA_PATH)
U_star = data['U_star']   # (N, 2, T) — u and v velocity
P_star = data['p_star']   # (N, T)
t_star = data['t'].flatten()  # (T,)
X_star = data['X_star']   # (N, 2)

N, _, T = U_star.shape
print(f"  Spatial points : {N:,}")
print(f"  Time steps     : {T}")

# ── Sample training data ──────────────────────────────────────────────────────
# Following Raissi et al. 2020:
# - N_data concentration observations (sparse, noisy)
# - N_col PDE collocation points (no data needed)

N_DATA = 5000    # concentration observations
N_COL  = 50000   # PDE collocation points

# Build full spatiotemporal dataset
x_all = np.tile(X_star[:, 0:1], (1, T)).flatten()
y_all = np.tile(X_star[:, 1:2], (1, T)).flatten()
t_all = np.tile(t_star, (N, 1)).flatten()

# Simulate concentration field c from velocity (passive scalar advected by flow)
# For the lecture we approximate c from the velocity magnitude as a proxy
# In the full Raissi 2020 paper, c is measured independently
c_all = np.sqrt(U_star[:, 0, :].flatten()**2 +
                U_star[:, 1, :].flatten()**2)
c_all = (c_all - c_all.min()) / (c_all.max() - c_all.min())  # normalise to [0,1]

# Random sample for data observations
rng = np.random.default_rng(SEED)
data_idx = rng.choice(len(x_all), N_DATA, replace=False)
x_data = x_all[data_idx]
y_data = y_all[data_idx]
t_data = t_all[data_idx]
c_data = c_all[data_idx]

# PDE collocation points — random in space-time domain
x_min, x_max = X_star[:, 0].min(), X_star[:, 0].max()
y_min, y_max = X_star[:, 1].min(), X_star[:, 1].max()
t_min, t_max = t_star.min(), t_star.max()

x_col = rng.uniform(x_min, x_max, N_COL)
y_col = rng.uniform(y_min, y_max, N_COL)
t_col = rng.uniform(t_min, t_max, N_COL)

print(f"\nTraining data:")
print(f"  Concentration observations : {N_DATA:,}")
print(f"  PDE collocation points     : {N_COL:,}")

# ── Normalise coordinates ─────────────────────────────────────────────────────
x_scale = x_max - x_min
y_scale = y_max - y_min
t_scale = t_max - t_min

def normalise(x, y, t):
    return (x - x_min) / x_scale, (y - y_min) / y_scale, (t - t_min) / t_scale

x_data_n, y_data_n, t_data_n = normalise(x_data, y_data, t_data)
x_col_n,  y_col_n,  t_col_n  = normalise(x_col,  y_col,  t_col)

# ── Initialise model ──────────────────────────────────────────────────────────
model = NS_PINN(n_hidden=9, n_neurons=20).to(device)
print(f"\nModel parameters: {model.n_params():,}")

# ── Training configuration ───────────────────────────────────────────────────
EPOCHS_ADAM  = 50000
EPOCHS_LBFGS = 1000
LR           = 1e-3
LAMBDA_PDE   = 1.0
LAMBDA_DATA  = 1.0
PRINT_EVERY  = 1000

history = {'adam_total': [], 'adam_pde': [], 'adam_data': [],
           'lbfgs_total': []}

# ── Phase 1: Adam ─────────────────────────────────────────────────────────────
print(f"\nPhase 1 — Adam | {EPOCHS_ADAM:,} epochs | lr={LR}")
print("-" * 60)

adam      = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(adam, step_size=10000, gamma=0.5)
t0 = time.time()

for epoch in range(1, EPOCHS_ADAM + 1):
    model.train()
    adam.zero_grad()
    total, losses = model.compute_loss(
        x_col_n, y_col_n, t_col_n,
        x_data_n, y_data_n, t_data_n, c_data,
        device,
        lambda_pde=LAMBDA_PDE, lambda_data=LAMBDA_DATA
    )
    total.backward()
    adam.step()
    scheduler.step()

    history['adam_total'].append(losses['total'])
    history['adam_pde'].append(losses['pde'])
    history['adam_data'].append(losses['data'])

    if epoch % PRINT_EVERY == 0:
        print(f"  epoch {epoch:6d} | total={losses['total']:.4e} | "
              f"pde={losses['pde']:.4e} | data={losses['data']:.4e} | "
              f"{time.time()-t0:.0f} s")

print(f"\nAdam complete | final loss = {history['adam_total'][-1]:.4e} | "
      f"time = {time.time()-t0:.0f} s")

# ── Phase 2: L-BFGS ───────────────────────────────────────────────────────────
print(f"\nPhase 2 — L-BFGS | {EPOCHS_LBFGS} iterations")
print("-" * 60)

lbfgs = optim.LBFGS(model.parameters(),
                     max_iter=EPOCHS_LBFGS,
                     tolerance_grad=1e-9,
                     tolerance_change=1e-11,
                     history_size=50,
                     line_search_fn='strong_wolfe')

def closure():
    lbfgs.zero_grad()
    total, losses = model.compute_loss(
        x_col_n, y_col_n, t_col_n,
        x_data_n, y_data_n, t_data_n, c_data,
        device,
        lambda_pde=LAMBDA_PDE, lambda_data=LAMBDA_DATA
    )
    total.backward()
    history['lbfgs_total'].append(losses['total'])
    return total

model.train()
lbfgs.step(closure)

print(f"L-BFGS complete | final loss = {history['lbfgs_total'][-1]:.4e} | "
      f"total time = {time.time()-t0:.0f} s")

# ── Save model and history ────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

np.savez(LOSS_PATH,
         adam_total  = np.array(history['adam_total']),
         adam_pde    = np.array(history['adam_pde']),
         adam_data   = np.array(history['adam_data']),
         lbfgs_total = np.array(history['lbfgs_total']))
print(f"Loss history saved: {LOSS_PATH}")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\nSanity check — evaluating on a few test points...")
model.eval()

x_test = np.array([1.0, 2.0, 3.0])
y_test = np.array([0.0, 0.5, -0.5])
t_test = np.array([2.0, 2.0, 2.0])
xtn, ytn, ttn = normalise(x_test, y_test, t_test)

def to_t(a):
    return torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)

with torch.no_grad():
    u_p, v_p, p_p, c_p = model(to_t(xtn), to_t(ytn), to_t(ttn))

print(f"  x={x_test} | u={u_p.cpu().numpy().flatten().round(3)}")
print(f"  x={x_test} | v={v_p.cpu().numpy().flatten().round(3)}")
print(f"\nTraining complete. Pre-trained model is ready for the lecture.")
print(f"Load in notebook with: ns_net.load_state_dict(torch.load('{MODEL_PATH}'))")
