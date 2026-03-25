"""
lecture_utils/plots.py
======================
Shared plotting functions for the MKM411 lecture notebooks.

Functions
---------
draw_network          — neural network architecture diagram
plot_sciml_landscape  — SciML method landscape overview
plot_method_comparison — method comparison summary tile diagram
plot_loss_comparison  — side-by-side ANN vs PINN loss diagrams
plot_cylinder_fields  — velocity and pressure field plots for cylinder wake
plot_vortex_street    — synthetic Von Kármán vortex street illustration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import UP_BLUE, UP_GOLD, ACCENT, FIGSIZE_WIDE


# ── Network diagram ───────────────────────────────────────────────────────────

def draw_network(ax, layer_sizes, title=None):
    """
    Draw a fully-connected neural network architecture diagram.

    Parameters
    ----------
    ax          : matplotlib Axes
    layer_sizes : list[int] — number of neurons per layer
    title       : str or None — axes title
    """
    n_layers   = len(layer_sizes)
    max_n      = max(layer_sizes)
    node_r     = 0.025
    colours    = [UP_BLUE] + ['steelblue'] * (n_layers - 2) + [UP_GOLD]
    labels     = (
        ['Input\n(' + str(layer_sizes[0]) + ')'] +
        ['×' + str(n) for n in layer_sizes[1:-1]] +
        ['Output\n(' + str(layer_sizes[-1]) + ')']
    )

    # Compute node positions
    positions = []
    for l, n in enumerate(layer_sizes):
        x = l / (n_layers - 1)
        ys = np.linspace(0.1, 0.9, n) if n > 1 else [0.5]
        positions.append([(x, y) for y in ys])

    # Draw connections
    for l in range(n_layers - 1):
        for (x1, y1) in positions[l]:
            for (x2, y2) in positions[l + 1]:
                ax.plot([x1, x2], [y1, y2], color='gray',
                        alpha=0.12, linewidth=0.5, zorder=1)

    # Draw nodes and layer labels
    for l, (layer_pos, colour, label) in enumerate(
            zip(positions, colours, labels)):
        for (x, y) in layer_pos:
            circle = plt.Circle((x, y), node_r, color=colour,
                                 zorder=5, linewidth=0)
            ax.add_patch(circle)
        mid_y = np.mean([y for _, y in layer_pos])
        ax.text(layer_pos[0][0], -0.05, label,
                ha='center', va='top', fontsize=8,
                color=colour, fontweight='bold')

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, color=UP_BLUE, pad=15)


# ── SciML landscape ───────────────────────────────────────────────────────────

def plot_sciml_landscape(highlight_pinn=True):
    """
    Visualise the Scientific Machine Learning landscape.

    Parameters
    ----------
    highlight_pinn : bool — annotate the PINN position
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    methods = {
        'FDM/FVM\n(classical)':           (0.05, 0.50, UP_BLUE,      200),
        'PINNs':                            (0.30, 0.50, UP_GOLD,      300),
        'Neural Operators\n(DeepONet, FNO)':(0.55, 0.70, 'steelblue', 250),
        'GNNs':                             (0.55, 0.30, 'teal',       200),
        'Data-driven\nSurrogates':          (0.75, 0.50, 'purple',     200),
        'Foundation\nModels':               (0.93, 0.50, ACCENT,       250),
    }

    for name, (x, y, colour, size) in methods.items():
        ax.scatter(x, y, s=size, color=colour, alpha=0.85, zorder=5)
        ax.text(x, y + 0.12, name, ha='center', fontsize=9,
                color=colour, fontweight='bold')

    # Axis arrows
    ax.annotate('', xy=(0.97, 0.12), xytext=(0.03, 0.12),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.50, 0.05, 'More data-driven →',
            ha='center', fontsize=10, color='gray', style='italic')

    ax.annotate('', xy=(0.03, 0.12), xytext=(0.97, 0.12),
                arrowprops=dict(arrowstyle='->', color=UP_BLUE, lw=1.5))
    ax.text(0.50, -0.02, '← More physics-constrained',
            ha='center', fontsize=10, color=UP_BLUE, style='italic')

    if highlight_pinn:
        ax.annotate('We are here\n(this lecture)',
                    xy=(0.30, 0.50),
                    xytext=(0.18, 0.80),
                    arrowprops=dict(arrowstyle='->', color=UP_GOLD, lw=2),
                    fontsize=10, color=UP_GOLD, fontweight='bold')

    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0)
    ax.axis('off')
    ax.set_title('The Scientific Machine Learning Landscape',
                 fontsize=13, color=UP_BLUE, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# ── Method comparison tiles ───────────────────────────────────────────────────

def plot_method_comparison():
    """
    Tile-based comparison of SciML methods for fluid dynamics.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

    methods = [
        (0.3, 5.5, 2.2, 1.8, UP_BLUE,
         'PINNs',
         'PDE in loss\nAutograd derivatives\nForward + Inverse'),
        (3.0, 5.5, 2.2, 1.8, 'steelblue',
         'DeepONet',
         'Branch + trunk\nOperator learning\nAny input function'),
        (5.7, 5.5, 2.2, 1.8, 'teal',
         'FNO',
         'Fourier convolution\nFast inference\nPeriodic domains'),
        (0.3, 2.8, 2.2, 1.8, 'purple',
         'GNNs',
         'Graph structure\nUnstructured meshes\nMessage passing'),
        (3.0, 2.8, 2.2, 1.8, ACCENT,
         'GANs',
         'Distribution learning\nTurbulence synthesis\nSuper-resolution'),
        (5.7, 2.8, 2.2, 1.8, UP_GOLD,
         'Foundation\nModels',
         'Pre-trained at scale\nFine-tunable\nAurora, Poseidon'),
        (8.2, 3.8, 1.6, 1.6, 'gray',
         'Classical\nCFD',
         'FDM / FVM\nPhysics-exact\nMature & reliable'),
    ]

    for x, y, w, h, col, title, desc in methods:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.1', linewidth=1.5,
            edgecolor=col,
            facecolor=col, alpha=0.15)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.25, title,
                ha='center', va='top', fontsize=10,
                fontweight='bold', color=col)
        ax.text(x + w / 2, y + h / 2 - 0.1, desc,
                ha='center', va='center', fontsize=7.5, color='black')

    ax.annotate('', xy=(8.2, 4.6), xytext=(7.9, 5.2),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(8.05, 5.35, 'Complementary\nnot competing',
            ha='center', fontsize=7, color='gray', style='italic')

    ax.set_title(
        'Scientific Machine Learning for Fluid Dynamics — Method Overview',
        fontsize=12, color=UP_BLUE, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


# ── ANN vs PINN loss diagram ──────────────────────────────────────────────────

def plot_loss_comparison():
    """
    Side-by-side illustration of ANN and PINN loss formulations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    np.random.seed(42)

    # ── ANN ───────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('ANN Loss — MSE against data',
                 fontsize=12, color=UP_BLUE, fontweight='bold')

    xs = np.random.uniform(1, 9, 8)
    ys = np.random.uniform(1.5, 6.5, 8)
    ax.scatter(xs, ys, s=80, color=UP_BLUE, zorder=5, label='Training data')

    x_pred = np.linspace(0.5, 9.5, 100)
    y_pred = 4 + 1.5 * np.sin(x_pred * 0.7) + 0.3 * np.sin(x_pred * 2)
    ax.plot(x_pred, y_pred, color=ACCENT, linewidth=2, label='Network output')

    for xi, yi in zip(xs, ys):
        y_net = 4 + 1.5 * np.sin(xi * 0.7) + 0.3 * np.sin(xi * 2)
        ax.plot([xi, xi], [yi, y_net], color='gray',
                linewidth=1, linestyle='--', alpha=0.7)

    ax.legend(fontsize=9, loc='upper left')
    ax.text(5, 0.2,
            r'$\mathcal{L} = \frac{1}{N}\sum(f_\theta - T_\mathrm{data})^2$',
            ha='center', fontsize=11, color=UP_BLUE)

    # ── PINN ──────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('PINN Loss — physics residuals, no data required',
                 fontsize=12, color=UP_GOLD, fontweight='bold')

    np.random.seed(7)
    xc = np.random.uniform(0.5, 9.5, 35)
    yc = np.random.uniform(0.5, 7.5, 35)
    ax.scatter(xc, yc, s=40, color='steelblue', alpha=0.6,
               marker='x', label='PDE collocation pts')

    Nw = 10
    xb = np.concatenate([np.linspace(0.3, 9.7, Nw),
                          np.linspace(0.3, 9.7, Nw),
                          np.full(6, 0.3), np.full(6, 9.7)])
    yb = np.concatenate([np.full(Nw, 0.3), np.full(Nw, 7.7),
                          np.linspace(0.3, 7.7, 6),
                          np.linspace(0.3, 7.7, 6)])
    ax.scatter(xb, yb, s=60, color=UP_GOLD, zorder=5, label='BC points')
    ax.scatter(np.linspace(0.5, 9.5, 10), np.full(10, 0.3),
               s=60, color=ACCENT, zorder=5, label='IC points (t=0)')

    ax.legend(fontsize=8, loc='upper left')
    ax.text(5, -0.5,
            r'$\mathcal{L} = \lambda_\mathrm{pde}\mathcal{L}_\mathrm{pde}'
            r'+ \lambda_\mathrm{bc}\mathcal{L}_\mathrm{bc}'
            r'+ \lambda_\mathrm{ic}\mathcal{L}_\mathrm{ic}$',
            ha='center', fontsize=10, color=UP_GOLD)

    plt.suptitle('Same architecture — different loss functions',
                 fontsize=11, color='gray', style='italic')
    plt.tight_layout()
    return fig


# ── Cylinder wake visualisation ───────────────────────────────────────────────

def plot_cylinder_fields(x, y, u, v, p, t_val=None):
    """
    Three-panel plot of cylinder wake velocity and pressure fields.

    Parameters
    ----------
    x, y    : ndarray (N,) — spatial coordinates
    u, v, p : ndarray (N,) — field values
    t_val   : float or None — evaluation time for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fields  = [u, v, p]
    labels  = ['u-velocity [m/s]', 'v-velocity [m/s]', 'Pressure [Pa]']
    cmaps   = ['RdBu_r', 'RdBu_r', 'viridis']
    titles  = ['u-velocity', 'v-velocity', 'Pressure']

    for ax, field, label, cmap, title in zip(
            axes, fields, labels, cmaps, titles):
        sc = ax.scatter(x, y, c=field, s=2, cmap=cmap)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        t_str = f'  (t = {t_val:.2f} s)' if t_val is not None else ''
        ax.set_title(f'{title}{t_str}', color=UP_BLUE, fontweight='bold')

    plt.suptitle('Von Kármán Vortex Street — PINN Prediction',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_vortex_street():
    """
    Synthetic Von Kármán vortex street illustration using superposed vortices.
    Used as a fallback when the Raissi dataset or pre-trained model is absent.
    """
    x = np.linspace(-1, 9, 300)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)

    # Superposition of point vortices — simplified illustration
    U = np.ones_like(X)
    V = np.zeros_like(X)
    vortex_xs = [1.5 + i * 1.2 for i in range(5)]
    for i, vx in enumerate(vortex_xs):
        sign   =  1 if i % 2 == 0 else -1
        vy_u   =  0.5
        vy_l   = -0.5
        strength = 0.35
        r2_u = (X - vx)**2 + (Y - vy_u)**2 + 0.01
        r2_l = (X - vx)**2 + (Y - vy_l)**2 + 0.01
        U += -sign * strength * (Y - vy_u) / r2_u
        V +=  sign * strength * (X - vx)   / r2_u
        U +=  sign * strength * (Y - vy_l) / r2_l
        V += -sign * strength * (X - vx)   / r2_l

    # Mask cylinder
    mask = X**2 + Y**2 < 0.26
    U[mask] = np.nan
    V[mask] = np.nan

    speed     = np.sqrt(U**2 + V**2)
    vorticity = (np.gradient(V, x, axis=1) - np.gradient(U, y, axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    im = axes[0].contourf(X, Y, speed, 30, cmap='hot')
    axes[0].streamplot(x, y, U, V, color='white',
                        linewidth=0.5, density=1.5, arrowsize=0.8)
    plt.colorbar(im, ax=axes[0], label='Speed [m/s]')
    axes[0].add_patch(plt.Circle((0, 0), 0.5, color='gray', zorder=5))
    axes[0].text(0, 0, 'Cylinder', ha='center', va='center',
                  fontsize=7, color='white')
    axes[0].set_aspect('equal')
    axes[0].set_title('Speed + Streamlines\n(synthetic illustration)',
                       color=UP_BLUE, fontweight='bold')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    im2 = axes[1].contourf(X, Y, vorticity, 30,
                             cmap='RdBu_r', vmin=-3, vmax=3)
    plt.colorbar(im2, ax=axes[1], label='Vorticity [1/s]')
    axes[1].add_patch(plt.Circle((0, 0), 0.5, color='gray', zorder=5))
    axes[1].set_aspect('equal')
    axes[1].set_title('Vorticity — alternating sign\n(Von Kármán shedding)',
                       color=UP_BLUE, fontweight='bold')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    plt.suptitle('Von Kármán Vortex Street at Re = 100 (synthetic)',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig
