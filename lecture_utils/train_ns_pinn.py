"""
train_ns_pinn.py
================
Overnight training script for the NS PINN on the cylinder wake dataset.
Forward surrogate approach: learns (u, v, p) from DNS data while enforcing
the incompressible Navier-Stokes equations at collocation points.

INSTRUCTOR USE ONLY — run this on Mjolnir before the lecture to generate
the pre-trained model file used in Lecture 2.

Usage
-----
    conda activate pinn-heat
    cd lectures/
    python lecture_utils/train_ns_pinn.py

Output
------
    data/ns_pinn_pretrained.pt         — saved model state dict
    data/ns_pinn_normalisation.npz     — normalisation constants (CRITICAL)
    data/ns_pinn_loss_history.npz      — training loss history for plotting

Estimated runtime
-----------------
    NVIDIA RTX A2000 (Mjolnir): ~1-2 hours
    CPU only: not recommended (8+ hours)

Dataset
-------
    data/cylinder_nektar_wake.mat
    Download from:
    https://github.com/maziarraissi/PINNs/tree/master/main/Data

Why forward surrogate, not inverse problem
------------------------------------------
    Raissi, Yazdani & Karniadakis (2020) solved an inverse problem: inferring
    (u, v, p) from sparse concentration measurements c(x,y,t). This requires
    a real independent passive scalar field. The cylinder_nektar_wake.mat
    dataset does not provide concentration — only (u, v, p) from DNS.
    The forward surrogate approach used here is well-posed with this data
    and demonstrates all the same core PINN concepts: autograd derivatives,
    composite physics + data loss, and two-phase optimisation.

Reference
---------
    Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
    Physics-informed neural networks. Journal of Computational Physics,
    378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lecture_utils.pinn   import NS_PINN
from lecture_utils.config import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device  : {device}")
print(f"PyTorch : {torch.__version__}")

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE, 'data', 'cylinder_nektar_wake.mat')
MODEL_PATH = os.path.join(BASE, 'data', 'ns_pinn_pretrained.pt')
NORM_PATH  = os.path.join(BASE, 'data', 'ns_pinn_normalisation.npz')
LOSS_PATH  = os.path.join(BASE, 'data', 'ns_pinn_loss_history.npz')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Download from: https://github.com/maziarraissi/PINNs/tree/master/main/Data"
    )

print(f"\nLoading dataset: {DATA_PATH}")
data   = scipy.io.loadmat(DATA_PATH)
U_star = data['U_star']
P_star = data['p_star']
t_star = data['t'].flatten()
X_star = data['X_star']

N, _, T = U_star.shape
print(f"  Spatial points : {N:,}")
print(f"  Time steps     : {T}")

x_all = np.tile(X_star[:, 0:1], (1, T)).flatten()
y_all = np.tile(X_star[:, 1:2], (1, T)).flatten()
t_all = np.tile(t_star,         (N, 1)).flatten()
u_all = U_star[:, 0, :].flatten()
v_all = U_star[:, 1, :].flatten()
p_all = P_star.flatten()

x_min, x_max = x_all.min(), x_all.max()
y_min, y_max = y_all.min(), y_all.max()
t_min, t_max = t_all.min(), t_all.max()
x_scale = x_max - x_min
y_scale = y_max - y_min
t_scale = t_max - t_min

def normalise(x, y, t):
    return ((x - x_min) / x_scale,
            (y - y_min) / y_scale,
            (t - t_min) / t_scale)

os.makedirs(os.path.dirname(NORM_PATH), exist_ok=True)
np.savez(NORM_PATH,
         x_min=x_min, x_max=x_max,
         y_min=y_min, y_max=y_max,
         t_min=t_min, t_max=t_max)
print(f"\nNormalisation constants saved: {NORM_PATH}")
print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
print(f"  y: [{y_min:.3f}, {y_max:.3f}]")
print(f"  t: [{t_min:.3f}, {t_max:.3f}]")

N_DATA = 50000
N_COL  = 50000
rng    = np.random.default_rng(SEED)

data_idx = rng.choice(len(x_all), N_DATA, replace=False)
x_data_n, y_data_n, t_data_n = normalise(
    x_all[data_idx], y_all[data_idx], t_all[data_idx])
u_data = u_all[data_idx]
v_data = v_all[data_idx]
p_data = p_all[data_idx]

x_col_n = rng.uniform(0, 1, N_COL)
y_col_n = rng.uniform(0, 1, N_COL)
t_col_n = rng.uniform(0, 1, N_COL)

print(f"\nTraining data:")
print(f"  (u, v, p) observations : {N_DATA:,}")
print(f"  PDE collocation points  : {N_COL:,}")

model = NS_PINN(n_hidden=9, n_neurons=20).to(device)
print(f"\nModel: 3 inputs -> [20x9] -> 3 outputs  |  parameters: {model.n_params():,}")

EPOCHS_ADAM  = 50000 
EPOCHS_LBFGS = 1000
LR           = 1e-3
LAMBDA_PDE   = 0.0 # start with pure data fitting
LAMBDA_PDE_MAX = 0.1   # ramped up to this over first 20k epochs
LAMBDA_DATA  = 1.0
PRINT_EVERY  = 2000

history = {'adam_total': [], 'adam_pde': [], 'adam_data': [], 'lbfgs_total': []}

print(f"\nPhase 1 - Adam | {EPOCHS_ADAM:,} epochs | lr={LR}")
print("-" * 60)

adam      = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(adam, step_size=10000, gamma=0.5)
t0 = time.time()

for epoch in range(1, EPOCHS_ADAM + 1):
    model.train()
    adam.zero_grad()
    lambda_pde = min(0.1, 0.1 * epoch / 20000)
    total, losses = model.compute_loss(
        x_col_n, y_col_n, t_col_n,
        x_data_n, y_data_n, t_data_n,
        u_data, v_data, p_data,
        device,
        x_scale=x_scale, y_scale=y_scale, t_scale=t_scale,
        lambda_pde=lambda_pde, lambda_data=LAMBDA_DATA,
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
              f"{time.time()-t0:.0f}s")

print(f"\nAdam complete | final loss = {history['adam_total'][-1]:.4e} | "
      f"time = {time.time()-t0:.0f}s")

print(f"\nPhase 2 - L-BFGS | {EPOCHS_LBFGS} iterations")
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
        x_data_n, y_data_n, t_data_n,
        u_data, v_data, p_data,
        device,
        x_scale=x_scale, y_scale=y_scale, t_scale=t_scale,
        lambda_pde=lambda_pde, lambda_data=LAMBDA_DATA,
    )
    total.backward()
    history['lbfgs_total'].append(losses['total'])
    return total

model.train()
lbfgs.step(closure)

print(f"L-BFGS complete | final loss = {history['lbfgs_total'][-1]:.4e} | "
      f"total time = {time.time()-t0:.0f}s")

torch.save(model.state_dict(), MODEL_PATH)
np.savez(LOSS_PATH,
         adam_total  = np.array(history['adam_total']),
         adam_pde    = np.array(history['adam_pde']),
         adam_data   = np.array(history['adam_data']),
         lbfgs_total = np.array(history['lbfgs_total']))
print(f"\nModel saved     : {MODEL_PATH}")
print(f"Loss history    : {LOSS_PATH}")

print("\nSanity check - comparing prediction vs DNS at 5 random test points...")
model.eval()

test_idx = rng.choice(len(x_all), 5, replace=False)
x_tn, y_tn, t_tn = normalise(x_all[test_idx], y_all[test_idx], t_all[test_idx])

def to_t(a):
    return torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)

with torch.no_grad():
    u_p, v_p, p_p = model(to_t(x_tn), to_t(y_tn), to_t(t_tn))

u_p = u_p.cpu().numpy().flatten()
v_p = v_p.cpu().numpy().flatten()

print(f"  {'':>4}  {'u_dns':>8}  {'u_pred':>8}  {'v_dns':>8}  {'v_pred':>8}")
for i in range(5):
    print(f"  {i:>4}  {u_all[test_idx[i]]:>8.4f}  {u_p[i]:>8.4f}  "
          f"{v_all[test_idx[i]]:>8.4f}  {v_p[i]:>8.4f}")

u_mae = np.mean(np.abs(u_p - u_all[test_idx]))
v_mae = np.mean(np.abs(v_p - v_all[test_idx]))
print(f"\n  MAE - u: {u_mae:.4f}  |  v: {v_mae:.4f}")
print(f"\nTraining complete. Pre-trained model ready for Lecture 2.")
