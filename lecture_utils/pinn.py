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
    Navier-Stokes PINN — forward surrogate for the 2D cylinder wake problem.

    Architecture follows Raissi et al. (2019):
    - 9 hidden layers × 20 neurons, tanh activation, Xavier initialisation
    - Input:  (x, y, t) — normalised to [0, 1]
    - Output: (u, v, p) — x-velocity, y-velocity, pressure

    Training strategy
    -----------------
    This is a FORWARD surrogate: the model learns from direct (u, v, p) observations
    from DNS/experimental data while simultaneously being constrained to satisfy the
    incompressible Navier-Stokes equations at collocation points.

    Composite loss:
        L = lambda_data * L_data  +  lambda_pde * L_pde

    where L_data = MSE on (u, v, p) at data points,
    and   L_pde  = NS residuals (x-momentum, y-momentum, continuity)
                   at collocation points — no data required there.

    Note on the Raissi 2020 inverse problem
    ----------------------------------------
    Raissi, Yazdani & Karniadakis (2020) solved a harder inverse problem:
    inferring (u, v, p) from sparse concentration measurements c(x,y,t) only,
    with no direct velocity or pressure observations. That requires a real
    independent passive scalar field, which the cylinder_nektar_wake.mat dataset
    does not provide. The forward surrogate approach used here is well-posed
    with the available data and demonstrates the same core PINN concepts.

    Reference
    ---------
    Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
    Physics-informed neural networks. Journal of Computational Physics, 378, 686-707.
    https://doi.org/10.1016/j.jcp.2018.10.045
    """

    def __init__(self, n_hidden=9, n_neurons=20):
        super().__init__()
        torch.manual_seed(SEED)
        self.n_hidden  = n_hidden
        self.n_neurons = n_neurons

        layers = [nn.Linear(3, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 3)]   # u, v, p
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
        x, y, t : normalised spatial coordinates and time in [0, 1]

        Returns
        -------
        u, v, p : Tensor (N, 1) each
        """
        inp = torch.cat([x, y, t], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def ns_residuals(self, x, y, t, rho=1.0, mu=0.01,
                     x_scale=1.0, y_scale=1.0, t_scale=1.0):
        """
        Compute three NS PDE residuals via automatic differentiation.

        Parameters
        ----------
        x, y, t : Tensor (N, 1) — normalised coordinates
        rho     : float — fluid density [kg/m³]
        mu      : float — dynamic viscosity [Pa·s]
        x_scale : float - range for input x
        y_scale : float - range for input y
        t_scale : float - range for input t

        Returns
        -------
        f_u   : Tensor (N,1) — x-momentum residual
        f_v   : Tensor (N,1) — y-momentum residual
        f_div : Tensor (N,1) — continuity residual
        """
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        u, v, p = self.forward(x, y, t)
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

        # ── Second-order derivatives ──────────────────────────────────────────
        u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, ones, create_graph=True)[0]

        v_xx = torch.autograd.grad(v_x, x, ones, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, ones, create_graph=True)[0]

        # ── PDE residuals ─────────────────────────────────────────────────────
        # x-momentum: rho(u_t/t_scale + u*u_x/x_scale  + v*u_y/y_scale) + p_x - mu*(u_xx/x_scale**2 + u_yy/y_scale**2) = 0
        # ── Apply chain rule corrections for normalised coordinates ──────────
        # autograd gives d/dx_hat — divide by scale to get d/dx (physical)
        f_u = (rho * (u_t / t_scale
                      + u * u_x / x_scale
                      + v * u_y / y_scale)
               + p_x / x_scale
               - mu * (u_xx / x_scale**2 + u_yy / y_scale**2))

        f_v = (rho * (v_t / t_scale
                      + u * v_x / x_scale
                      + v * v_y / y_scale)
               + p_y / y_scale
               - mu * (v_xx / x_scale**2 + v_yy / y_scale**2))

        f_div = u_x / x_scale + v_y / y_scale

        return f_u, f_v, f_div

    def compute_loss(self, x_c, y_c, t_c,
                     x_data, y_data, t_data,
                     u_data, v_data, p_data,
                     device, rho=1.0, mu=0.01,
                     x_scale=1.0, y_scale=1.0, t_scale=1.0,
                     lambda_pde=1.0, lambda_data=1.0):
        """
        Compute composite PINN loss.

        Parameters
        ----------
        x_c, y_c, t_c          : ndarray — PDE collocation points (normalised)
        x_data, y_data, t_data : ndarray — data observation points (normalised)
        u_data, v_data, p_data : ndarray — observed field values at data points
        device                 : str
        lambda_pde, lambda_data : float — loss weights
        x_scale : float - range for input x
        y_scale : float - range for input y
        t_scale : float - range for input t

        Returns
        -------
        total  : scalar Tensor
        losses : dict with individual loss values
        """
        def to_t(a):
            return torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)

        # ── PDE loss at collocation points ────────────────────────────────────
        xc = to_t(x_c); yc = to_t(y_c); tc = to_t(t_c)
        f_u, f_v, f_div = self.ns_residuals(xc, yc, tc, rho, mu, x_scale, y_scale, t_scale)
        L_pde = (torch.mean(f_u**2) + torch.mean(f_v**2) +
                 torch.mean(f_div**2))

        # ── Data loss on all three fields ─────────────────────────────────────
        xd = to_t(x_data); yd = to_t(y_data); td = to_t(t_data)
        ud = to_t(u_data); vd = to_t(v_data); pd_ = to_t(p_data)
        u_pred, v_pred, p_pred = self.forward(xd, yd, td)
        L_data = (torch.mean((u_pred - ud)**2) +
                  torch.mean((v_pred - vd)**2) +
                  torch.mean((p_pred - pd_)**2))

        total = lambda_pde * L_pde + lambda_data * L_data

        return total, {
            'pde':   L_pde.item(),
            'data':  L_data.item(),
            'total': total.item(),
        }

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
