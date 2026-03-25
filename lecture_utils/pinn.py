"""
lecture_utils/pinn.py
=====================
PINN architectures for the lecture notebooks.

MinimalHeatPINN  — 1D heat equation PINN for autograd demonstration (Lecture 2)
NS_PINN          — 2D Navier-Stokes PINN for cylinder wake problem (Lecture 2)
                   Follows Raissi et al. (2020) architecture: 9 × 20, tanh
"""

import torch
import torch.nn as nn
from .config import SEED


class MinimalHeatPINN(nn.Module):
    """
    Minimal 1D heat equation PINN for lecture demonstration.

    Illustrates how autograd computes PDE derivatives through the network.
    Not used for the project — see project utils/models.py for the full 2D version.

    Governing equation: alpha * T_xx - T_t = 0

    Input : (x, t) — 2 inputs
    Output: T — scalar temperature

    Architecture: 2 → 20 → 20 → 1  (tanh)
    """

    def __init__(self, n_hidden=2, n_neurons=20):
        super().__init__()
        torch.manual_seed(SEED)
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

    def pde_residual(self, x, t, alpha=1e-4):
        """
        Compute 1D heat equation residual via automatic differentiation.

        PDE: alpha * d²T/dx² - dT/dt = 0

        Parameters
        ----------
        x, t  : Tensor (N, 1) — spatial and temporal coordinates
        alpha : float — thermal diffusivity [m²/s]

        Returns
        -------
        residual : Tensor (N, 1) — PDE residual at each point
        """
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        T    = self.forward(x, t)
        ones = torch.ones_like(T)

        T_t  = torch.autograd.grad(T,   t, ones, create_graph=True)[0]
        T_x  = torch.autograd.grad(T,   x, ones, create_graph=True)[0]
        T_xx = torch.autograd.grad(T_x, x, ones, create_graph=True)[0]

        return alpha * T_xx - T_t


class NS_PINN(nn.Module):
    """
    Navier-Stokes PINN for the 2D cylinder wake problem.

    Follows the architecture of Raissi, Yazdani & Karniadakis (2020):
    - 9 hidden layers × 20 neurons, tanh activation, Xavier initialisation
    - Input: (x, y, t)
    - Output: (u, v, p, c) — x-velocity, y-velocity, pressure, concentration

    The PINN solves the INVERSE problem: given sparse observations of c(x,y,t),
    infer the full velocity (u, v) and pressure (p) fields by enforcing:
      1. Navier-Stokes x-momentum
      2. Navier-Stokes y-momentum
      3. Continuity (incompressibility)
      4. Advection-diffusion for concentration c

    Reference
    ---------
    Raissi, M., Yazdani, A., & Karniadakis, G.E. (2020).
    Hidden fluid mechanics: Learning velocity and pressure fields from
    flow visualizations. Science, 367(6481), 1026-1030.
    https://doi.org/10.1126/science.aaw4741
    """

    def __init__(self, n_hidden=9, n_neurons=20):
        super().__init__()
        torch.manual_seed(SEED)
        self.n_hidden  = n_hidden
        self.n_neurons = n_neurons

        layers = [nn.Linear(3, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 4)]   # u, v, p, c
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, y, t):
        """
        Forward pass.

        Parameters (all Tensor (N, 1))
        ----------
        x, y, t : spatial coordinates and time

        Returns
        -------
        u, v, p, c : Tensor (N, 1) each — four field predictions
        """
        inp = torch.cat([x, y, t], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    def ns_residuals(self, x, y, t, rho=1.0, mu=0.01, kappa=0.01):
        """
        Compute all four PDE residuals via automatic differentiation.

        Parameters
        ----------
        x, y, t : Tensor (N, 1) — coordinates (must NOT have requires_grad set
                  externally — this method sets requires_grad internally)
        rho     : float — fluid density [kg/m³]
        mu      : float — dynamic viscosity [Pa·s]
        kappa   : float — scalar diffusivity [m²/s]

        Returns
        -------
        f_u   : Tensor (N,1) — x-momentum residual
        f_v   : Tensor (N,1) — y-momentum residual
        f_div : Tensor (N,1) — continuity (divergence) residual
        f_adv : Tensor (N,1) — advection-diffusion residual
        """
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        u, v, p, c = self.forward(x, y, t)
        ones = torch.ones_like(u)

        # ── First-order derivatives ───────────────────────────────────────────
        u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, ones, create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, ones, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, ones, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, ones, create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, ones, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, ones, create_graph=True)[0]

        c_t = torch.autograd.grad(c, t, ones, create_graph=True)[0]
        c_x = torch.autograd.grad(c, x, ones, create_graph=True)[0]
        c_y = torch.autograd.grad(c, y, ones, create_graph=True)[0]

        # ── Second-order derivatives ──────────────────────────────────────────
        u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, ones, create_graph=True)[0]

        v_xx = torch.autograd.grad(v_x, x, ones, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, ones, create_graph=True)[0]

        c_xx = torch.autograd.grad(c_x, x, ones, create_graph=True)[0]
        c_yy = torch.autograd.grad(c_y, y, ones, create_graph=True)[0]

        # ── PDE residuals ─────────────────────────────────────────────────────
        # x-momentum: rho(u_t + u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) = 0
        f_u = rho * (u_t + u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)

        # y-momentum: rho(v_t + u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) = 0
        f_v = rho * (v_t + u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)

        # Continuity: u_x + v_y = 0
        f_div = u_x + v_y

        # Advection-diffusion: c_t + u*c_x + v*c_y - kappa*(c_xx + c_yy) = 0
        f_adv = c_t + u * c_x + v * c_y - kappa * (c_xx + c_yy)

        return f_u, f_v, f_div, f_adv

    def compute_loss(self, x_c, y_c, t_c,
                     x_data, y_data, t_data, c_data,
                     device, rho=1.0, mu=0.01, kappa=0.01,
                     lambda_pde=1.0, lambda_data=1.0):
        """
        Compute composite PINN loss.

        Parameters
        ----------
        x_c, y_c, t_c       : ndarray — PDE collocation points
        x_data, y_data, t_data, c_data : ndarray — observed concentration data
        device               : str
        lambda_pde, lambda_data : float — loss weights

        Returns
        -------
        total  : scalar Tensor
        losses : dict with individual loss values
        """
        def to_t(a):
            return torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)

        # ── PDE loss ──────────────────────────────────────────────────────────
        xc = to_t(x_c); yc = to_t(y_c); tc = to_t(t_c)
        f_u, f_v, f_div, f_adv = self.ns_residuals(xc, yc, tc, rho, mu, kappa)
        L_pde = (torch.mean(f_u**2) + torch.mean(f_v**2) +
                 torch.mean(f_div**2) + torch.mean(f_adv**2))

        # ── Data loss (concentration observations only) ────────────────────────
        xd = to_t(x_data); yd = to_t(y_data); td = to_t(t_data)
        cd = to_t(c_data)
        _, _, _, c_pred = self.forward(xd, yd, td)
        L_data = torch.mean((c_pred - cd)**2)

        total = lambda_pde * L_pde + lambda_data * L_data

        return total, {
            'pde':  L_pde.item(),
            'data': L_data.item(),
            'total': total.item(),
        }

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
