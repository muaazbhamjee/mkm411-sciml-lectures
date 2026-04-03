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

def plot_sciml_landscape(highlight_pinn=True, highlight_fdm_fvm=True):
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
                    xytext=(0.35, 0.80),
                    arrowprops=dict(arrowstyle='->', color=UP_GOLD, lw=2),
                    fontsize=10, color=UP_GOLD, fontweight='bold')
        
    if highlight_fdm_fvm:
        ax.annotate('You are learning this \n(in MKM 411)',
                    xy=(0.05, 0.50),
                    xytext=(0.09, 0.80),
                    arrowprops=dict(arrowstyle='->', color=UP_BLUE, lw=2),
                    fontsize=10, color=UP_BLUE, fontweight='bold')

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


# ═══════════════════════════════════════════════════════════════════════════════
# THEORY FIGURES — added for expanded Lecture 1
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Feedforward pass ────────────────────────────────────────────────

def plot_feedforward():
    """
    Annotated feedforward pass diagram showing matrix dimensions at each layer.
    Data flows left to right. Each layer annotated with W, b, and activation.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis('off')

    layer_sizes = [3, 4, 4, 2]
    layer_labels = ['Input\n$\mathbf{x}$', 'Hidden 1\n$\mathbf{a}^{(1)}$',
                    'Hidden 2\n$\mathbf{a}^{(2)}$', 'Output\n$\hat{\mathbf{y}}$']
    dim_labels   = ['$n_0=3$', '$n_1=4$', '$n_2=4$', '$n_3=2$']
    xs = [1.2, 4.2, 7.8, 11.2]
    colours = [UP_BLUE, 'steelblue', 'steelblue', UP_GOLD]
    node_r = 0.28

    positions = []
    for x, n in zip(xs, layer_sizes):
        ys = np.linspace(1.5, 5.5, n)
        positions.append(list(zip([x]*n, ys)))

    # Connections
    for l in range(len(positions) - 1):
        for (x1, y1) in positions[l]:
            for (x2, y2) in positions[l+1]:
                ax.plot([x1, x2], [y1, y2], color='gray',
                        alpha=0.2, linewidth=0.8, zorder=1)

    # Nodes
    for l, (layer_pos, colour) in enumerate(zip(positions, colours)):
        for (x, y) in layer_pos:
            circle = plt.Circle((x, y), node_r, color=colour,
                                 zorder=5, alpha=0.9)
            ax.add_patch(circle)
        # Layer label
        ax.text(layer_pos[0][0], 0.9, layer_labels[l],
                ha='center', fontsize=9, color=colour, fontweight='bold')
        ax.text(layer_pos[0][0], 0.3, dim_labels[l],
                ha='center', fontsize=8, color='gray')

    # Weight matrix annotations between layers
    w_labels = [
        '$W^{(1)} \in \mathbb{R}^{4 \times 3}$\n$\mathbf{b}^{(1)} \in \mathbb{R}^4$',
        '$W^{(2)} \in \mathbb{R}^{4 \times 4}$\n$\mathbf{b}^{(2)} \in \mathbb{R}^4$',
        '$W^{(3)} \in \mathbb{R}^{2 \times 4}$\n$\mathbf{b}^{(3)} \in \mathbb{R}^2$',
    ]
    w_xs = [2.7, 6.0, 9.5]
    for wx, wl in zip(w_xs, w_labels):
        ax.text(wx, 6.3, wl, ha='center', fontsize=8,
                color=UP_BLUE,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EEF4FF',
                          edgecolor=UP_BLUE, alpha=0.8))

    # Activation annotation
    for wx in w_xs:
        ax.text(wx, 1.55, '$\sigma(\cdot)$', ha='center', fontsize=9,
                color=UP_GOLD, fontweight='bold')

    # Forward arrows
    for i in range(len(xs) - 1):
        ax.annotate('', xy=(xs[i+1] - 0.35, 3.5), xytext=(xs[i] + 0.35, 3.5),
                    arrowprops=dict(arrowstyle='->', color=UP_BLUE, lw=2))

    ax.text(6.5, 6.85,
            r'$\mathbf{a}^{(l)} = \sigma\!\left(W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$',
            ha='center', fontsize=12, color=UP_BLUE, fontweight='bold')

    ax.set_title('Feedforward Pass — Information Flow Left to Right',
                 fontsize=12, color=UP_BLUE, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


# ── Figure 2: Xavier initialisation ──────────────────────────────────────────

def plot_xavier_initialisation():
    """
    Signal variance across layers with and without Xavier initialisation.
    Shows collapse (too small), explosion (too large), and stable (Xavier).
    """
    import torch
    import torch.nn as nn

    def signal_variance(n_layers, n_neurons, init='xavier'):
        torch.manual_seed(42)
        x = torch.randn(1000, n_neurons)
        variances = [x.var().item()]
        for _ in range(n_layers):
            layer = nn.Linear(n_neurons, n_neurons, bias=False)
            if init == 'xavier':
                nn.init.xavier_normal_(layer.weight)
            elif init == 'too_small':
                nn.init.normal_(layer.weight, std=0.01)
            elif init == 'too_large':
                nn.init.normal_(layer.weight, std=2.0)
            with torch.no_grad():
                x = torch.tanh(layer(x))
            variances.append(x.var().item())
        return variances

    n_layers  = 15
    n_neurons = 64
    layers    = list(range(n_layers + 1))

    var_xavier    = signal_variance(n_layers, n_neurons, 'xavier')
    var_too_small = signal_variance(n_layers, n_neurons, 'too_small')
    var_too_large = signal_variance(n_layers, n_neurons, 'too_large')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Variance across layers ────────────────────────────────────────────────
    axes[0].plot(layers, var_xavier,    color=UP_BLUE,  lw=2.5,
                 label='Xavier init — stable')
    axes[0].plot(layers, var_too_small, color=ACCENT,   lw=2.0, ls='--',
                 label='Too small (std=0.01) — vanishes')
    axes[0].plot(layers, var_too_large, color=UP_GOLD,  lw=2.0, ls='-.',
                 label='Too large (std=2.0) — explodes')
    axes[0].set_xlabel('Layer depth', fontsize=11)
    axes[0].set_ylabel('Activation variance', fontsize=11)
    axes[0].set_title('Signal Variance Across Layers', fontsize=11,
                       color=UP_BLUE, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # ── Xavier derivation panel ───────────────────────────────────────────────
    axes[1].axis('off')
    derivation = (
        "Xavier / Glorot Initialisation\n\n"
        "Goal: preserve signal variance across layers\n\n"
        "For layer $l$ with $n_{in}$ inputs and $n_{out}$ outputs:\n\n"
        r"    $\mathrm{Var}(z^{(l)}) = n_{in} \cdot \mathrm{Var}(w) \cdot \mathrm{Var}(a^{(l-1)})$" + "\n\n"
        r"Set $\mathrm{Var}(w) = \frac{2}{n_{in} + n_{out}}$  so that:" + "\n\n"
        r"    $\mathrm{Var}(z^{(l)}) \approx \mathrm{Var}(a^{(l-1)})$" + "\n\n"
        "In practice: sample weights from\n\n"
        r"    $w \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{in}+n_{out}}\right)$" + "\n\n"
        "This is the default in the project — see models.py\n"
        "xavier_normal_() in PyTorch."
    )
    axes[1].text(0.05, 0.95, derivation, transform=axes[1].transAxes,
                  fontsize=10, va='top', ha='left', color=UP_BLUE,
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#EEF4FF',
                            edgecolor=UP_BLUE, alpha=0.85),
                  linespacing=1.6)
    axes[1].set_title('Why Xavier Works', fontsize=11,
                       color=UP_BLUE, fontweight='bold')

    plt.suptitle('Weight Initialisation — Getting the Scale Right',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Figure 3: Backpropagation ─────────────────────────────────────────────────

def plot_backpropagation():
    """
    Backpropagation information flow diagram.
    Mirrors the feedforward figure — gradients flow right to left.
    Shows delta error signals and weight gradient computation at each layer.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    def draw_pass(ax, forward=True):
        layer_sizes = [3, 4, 4, 2]
        xs = [1.2, 4.2, 7.8, 11.2]
        colours = [UP_BLUE, 'steelblue', 'steelblue', UP_GOLD]
        node_r  = 0.28

        positions = []
        for x, n in zip(xs, layer_sizes):
            ys = np.linspace(1.2, 4.8, n)
            positions.append(list(zip([x]*n, ys)))

        # Connections
        conn_colour = 'gray' if forward else ACCENT
        for l in range(len(positions) - 1):
            for (x1, y1) in positions[l]:
                for (x2, y2) in positions[l+1]:
                    ax.plot([x1, x2], [y1, y2],
                            color=conn_colour, alpha=0.15,
                            linewidth=0.8, zorder=1)

        # Nodes
        for l, (layer_pos, colour) in enumerate(zip(positions, colours)):
            for (x, y) in layer_pos:
                circle = plt.Circle((x, y), node_r, color=colour,
                                     zorder=5, alpha=0.9)
                ax.add_patch(circle)

        if forward:
            # Forward annotations
            labels = ['$\mathbf{x}$', '$\mathbf{a}^{(1)}$',
                      '$\mathbf{a}^{(2)}$', '$\hat{\mathbf{y}}$']
            for x, label in zip(xs, labels):
                ax.text(x, 0.6, label, ha='center', fontsize=10,
                        color=UP_BLUE, fontweight='bold')
            for i in range(len(xs) - 1):
                ax.annotate('', xy=(xs[i+1]-0.35, 3.0),
                            xytext=(xs[i]+0.35, 3.0),
                            arrowprops=dict(arrowstyle='->', color=UP_BLUE, lw=2))
            ax.text(6.5, 5.4,
                    r'Forward: $\mathbf{a}^{(l)} = \sigma(W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$',
                    ha='center', fontsize=10, color=UP_BLUE, fontweight='bold')
            ax.text(6.5, 0.1, 'Step 1 — Forward pass: compute all activations',
                    ha='center', fontsize=9, color='gray', style='italic')

        else:
            # Backprop annotations
            delta_labels = ['', '$\delta^{(1)}$',
                            '$\delta^{(2)}$',
                            '$\delta^{(3)}$']
            for x, label in zip(xs, delta_labels):
                if label:
                    ax.text(x, 0.6, label, ha='center', fontsize=10,
                            color=ACCENT, fontweight='bold')

            # Backward arrows
            for i in range(len(xs)-1, 0, -1):
                ax.annotate('', xy=(xs[i-1]+0.35, 3.0),
                            xytext=(xs[i]-0.35, 3.0),
                            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2.5))

            ax.text(6.5, 5.4,
                    r'Backward: $\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma^\prime(\mathbf{z}^{(l)})$',
                    ha='center', fontsize=10, color=ACCENT, fontweight='bold')

            # Gradient annotations
            grad_xs = [2.7, 6.0, 9.5]
            for gx in grad_xs:
                ax.text(gx, 5.0,
                        r'$\nabla_{W^{(l)}}\mathcal{L} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T$',
                        ha='center', fontsize=8, color=UP_GOLD,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF8E1',
                                  edgecolor=UP_GOLD, alpha=0.85))
            ax.text(6.5, 0.1,
                    'Step 2 — Backward pass: propagate error signals, compute weight gradients',
                    ha='center', fontsize=9, color='gray', style='italic')

        ax.set_xlim(0, 13); ax.set_ylim(0, 6)
        ax.axis('off')

    draw_pass(axes[0], forward=True)
    draw_pass(axes[1], forward=False)

    plt.suptitle('Backpropagation — Forward then Backward Pass',
                 fontsize=13, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Figure 4: Bias-variance ───────────────────────────────────────────────────

def plot_bias_variance():
    """
    Bias-variance trade-off diagram.
    Left panel: train vs val loss curves showing underfitting, good fit, overfitting.
    Right panel: classic U-shaped test error vs model complexity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Panel 1: Loss curves ──────────────────────────────────────────────────
    epochs = np.linspace(0, 100, 300)

    train_loss = 2.0 * np.exp(-0.04 * epochs) + 0.05
    val_loss_good = (2.0 * np.exp(-0.03 * epochs) + 0.25 +
                     0.002 * epochs * np.exp(-0.02 * epochs))
    val_loss_overfit = (2.0 * np.exp(-0.03 * epochs) + 0.15 +
                        0.008 * epochs)
    val_loss_overfit = np.minimum(val_loss_overfit, 3.0)

    axes[0].plot(epochs, train_loss,      color=UP_BLUE, lw=2.5, label='Training loss')
    axes[0].plot(epochs, val_loss_good,   color=UP_GOLD, lw=2.0, ls='--',
                 label='Val loss — good fit')
    axes[0].plot(epochs, val_loss_overfit, color=ACCENT, lw=2.0, ls='-.',
                 label='Val loss — overfitting')

    # Annotations
    axes[0].axvline(60, color='gray', lw=1, ls=':', alpha=0.7)
    axes[0].text(61, 1.8, 'Overfitting\nbegins', fontsize=8,
                  color='gray', style='italic')
    axes[0].annotate('', xy=(62, 1.5), xytext=(55, 1.5),
                      arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Train vs Validation Loss', fontsize=11,
                       color=UP_BLUE, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 2.5)

    # ── Panel 2: Bias-variance decomposition ──────────────────────────────────
    complexity = np.linspace(0.5, 10, 200)
    bias2    = 2.5 * np.exp(-0.5 * complexity)
    variance = 0.05 * np.exp(0.4 * complexity)
    total    = bias2 + variance + 0.1

    axes[1].plot(complexity, bias2,    color=UP_BLUE, lw=2.0, ls='--',
                 label='Bias²  (underfitting)')
    axes[1].plot(complexity, variance, color=ACCENT,  lw=2.0, ls='--',
                 label='Variance  (overfitting)')
    axes[1].plot(complexity, total,    color=UP_GOLD,  lw=2.5,
                 label='Total error (bias² + variance + noise)')

    # Mark optimal
    opt_idx = np.argmin(total)
    axes[1].axvline(complexity[opt_idx], color='gray', lw=1.5, ls=':')
    axes[1].scatter([complexity[opt_idx]], [total[opt_idx]],
                     s=120, color=UP_GOLD, zorder=6)
    axes[1].text(complexity[opt_idx] + 0.2, total[opt_idx] + 0.05,
                  'Optimal\ncomplexity', fontsize=8, color='gray',
                  style='italic')

    # Region labels
    axes[1].text(1.5, 2.2, 'Underfitting\n(high bias)',
                  ha='center', fontsize=8, color=UP_BLUE, alpha=0.8)
    axes[1].text(8.5, 2.2, 'Overfitting\n(high variance)',
                  ha='center', fontsize=8, color=ACCENT, alpha=0.8)

    axes[1].set_xlabel('Model complexity', fontsize=11)
    axes[1].set_ylabel('Error', fontsize=11)
    axes[1].set_title('Bias-Variance Trade-off', fontsize=11,
                       color=UP_BLUE, fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 2.8)

    plt.suptitle('Overfitting — Reading the Validation Loss Curve',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PANORAMA FIGURES — one per SciML method, individual cells
# ═══════════════════════════════════════════════════════════════════════════════

def _method_box(ax, title, subtitle, colour, body_lines,
                x=0.05, y=0.92, width=0.9, pad=0.04):
    """Helper: draw a titled info box on an axis."""
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Title banner
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y - 0.12), width, 0.14,
        boxstyle='round,pad=0.01', color=colour, alpha=0.9, zorder=3))
    ax.text(x + width/2, y - 0.04, title,
            ha='center', va='center', fontsize=13,
            fontweight='bold', color='white', zorder=4)
    ax.text(x + width/2, y - 0.10, subtitle,
            ha='center', va='center', fontsize=8.5,
            color='white', alpha=0.9, zorder=4, style='italic')
    # Body
    body_y = y - 0.18
    for line in body_lines:
        ax.text(x + pad, body_y, line,
                ha='left', va='top', fontsize=9.5, color='black')
        body_y -= 0.085


def plot_deeponet():
    """DeepONet architecture schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Diagram ───────────────────────────────────────────────────────────────
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('DeepONet Architecture', fontsize=11, color=UP_BLUE,
                  fontweight='bold')

    # Branch net
    for i, y in enumerate(np.linspace(1.5, 5.5, 4)):
        ax.add_patch(plt.Circle((1.5, y), 0.22, color=UP_BLUE, alpha=0.8))
    ax.add_patch(mpatches.FancyBboxPatch((0.6, 1.0), 1.8, 5.2,
                  boxstyle='round,pad=0.1', fill=False,
                  edgecolor=UP_BLUE, lw=1.5, linestyle='--'))
    ax.text(1.5, 0.5, 'Branch net\n$u(x_1),\ldots,u(x_m)$',
            ha='center', fontsize=9, color=UP_BLUE, fontweight='bold')

    # Trunk net
    for i, y in enumerate(np.linspace(1.5, 5.5, 4)):
        ax.add_patch(plt.Circle((4.5, y), 0.22, color='steelblue', alpha=0.8))
    ax.add_patch(mpatches.FancyBboxPatch((3.6, 1.0), 1.8, 5.2,
                  boxstyle='round,pad=0.1', fill=False,
                  edgecolor='steelblue', lw=1.5, linestyle='--'))
    ax.text(4.5, 0.5, 'Trunk net\nquery point $y$',
            ha='center', fontsize=9, color='steelblue', fontweight='bold')

    # Dot product
    ax.add_patch(plt.Circle((7.0, 3.5), 0.45, color=UP_GOLD, alpha=0.9, zorder=5))
    ax.text(7.0, 3.5, '$\cdot$', ha='center', va='center',
            fontsize=20, color='white', fontweight='bold', zorder=6)
    ax.text(7.0, 0.5, 'Dot product\n$\sum_k b_k t_k$',
            ha='center', fontsize=9, color=UP_GOLD, fontweight='bold')

    # Output
    ax.add_patch(mpatches.FancyBboxPatch((8.3, 3.0), 1.2, 1.0,
                  boxstyle='round,pad=0.1', color=ACCENT, alpha=0.8, zorder=5))
    ax.text(8.9, 3.5, '$\mathcal{G}(u)(y)$',
            ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', zorder=6)

    # Arrows
    for src, dst in [((2.4, 3.5), (6.55, 3.5)), ((5.4, 3.5), (6.55, 3.5)),
                     ((7.45, 3.5), (8.3, 3.5))]:
        ax.annotate('', xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # ── Info box ──────────────────────────────────────────────────────────────
    _method_box(axes[1], 'DeepONet',
                'Deep Operator Network — Lu et al. (2021)',
                UP_BLUE, [
                    '• Learns a mapping between function spaces',
                    '  $\mathcal{G}: u(\cdot) \mapsto s(\cdot)$',
                    '',
                    '• Branch net encodes the input function $u$',
                    '  evaluated at $m$ fixed sensor points',
                    '',
                    '• Trunk net encodes the query location $y$',
                    '',
                    '• Output = dot product of branch and trunk',
                    '  → one forward pass gives $s$ at any $y$',
                    '',
                    '• CFD use: parametric PDE solution operators,',
                    '  generalises across initial conditions',
                ])

    plt.suptitle('Neural Operator — DeepONet', fontsize=12,
                 color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_fno():
    """Fourier Neural Operator architecture schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]; ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('Fourier Neural Operator — One Layer', fontsize=11,
                  color=UP_BLUE, fontweight='bold')

    def block(ax, x, y, w, h, colour, label, sublabel=''):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                      boxstyle='round,pad=0.1', color=colour, alpha=0.85, zorder=3))
        ax.text(x + w/2, y + h/2 + 0.1, label,
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.3, sublabel,
                    ha='center', va='center', fontsize=7.5,
                    color='white', alpha=0.9, zorder=4)

    block(ax, 0.3, 2.2, 1.6, 1.6, UP_BLUE,    'Input\n$v_l$')
    block(ax, 2.5, 3.5, 2.0, 1.2, 'steelblue', 'FFT',       '$\mathcal{F}$')
    block(ax, 5.0, 3.5, 2.0, 1.2, UP_GOLD,     'Linear $R$', 'low freq only')
    block(ax, 7.5, 3.5, 2.0, 1.2, 'steelblue', 'IFFT',      '$\mathcal{F}^{-1}$')
    block(ax, 2.5, 1.0, 2.0, 1.2, 'teal',      'Linear $W$', '(local)')

    # Sum node
    ax.add_patch(plt.Circle((10.0, 3.0), 0.38, color=UP_GOLD, alpha=0.9, zorder=5))
    ax.text(10.0, 3.0, '+', ha='center', va='center',
            fontsize=16, color='white', fontweight='bold', zorder=6)

    block(ax, 10.8, 2.2, 1.0, 1.6, ACCENT, '$\sigma$\n$v_{l+1}$')

    # Arrows
    arrows = [
        ((1.9, 3.0), (2.5, 4.1)),   # input → FFT
        ((4.5, 4.1), (5.0, 4.1)),   # FFT → R
        ((7.0, 4.1), (7.5, 4.1)),   # R → IFFT
        ((9.5, 4.1), (9.62, 3.38)), # IFFT → sum
        ((1.9, 3.0), (2.5, 1.6)),   # input → W
        ((4.5, 1.6), (9.62, 2.62)), # W → sum
        ((10.38, 3.0), (10.8, 3.0)),# sum → output
    ]
    for src, dst in arrows:
        ax.annotate('', xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.3))

    ax.text(6.0, 0.3,
            'Key: linear transform in Fourier space = global convolution',
            ha='center', fontsize=8.5, color='gray', style='italic')

    _method_box(axes[1], 'Fourier Neural Operator (FNO)',
                'Li et al. (2021)',
                'steelblue', [
                    '• Learns a convolution kernel in Fourier space',
                    '  → equivalent to global spatial convolution',
                    '',
                    '• Operates on discretised functions on a grid',
                    '  — resolution-invariant at inference',
                    '',
                    '• Each FNO layer:',
                    '  FFT → linear transform (low freq) → IFFT',
                    '  + local linear transform (bypass)',
                    '',
                    '• Very fast at inference — no PDE solve',
                    '',
                    '• CFD use: turbulence, weather prediction,',
                    '  Navier-Stokes surrogate models',
                ])

    plt.suptitle('Neural Operator — Fourier Neural Operator (FNO)',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_gnn():
    """Graph Neural Network message passing schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('GNN — Message Passing on a CFD Mesh',
                 fontsize=11, color=UP_BLUE, fontweight='bold')

    # Mesh nodes
    nodes = {
        'A': (3.0, 5.5), 'B': (5.5, 5.5), 'C': (8.0, 5.5),
        'D': (2.0, 3.5), 'E': (5.0, 3.5), 'F': (7.5, 3.5),
        'G': (3.5, 1.5), 'H': (6.5, 1.5),
    }
    edges = [('A','B'),('B','C'),('D','E'),('E','F'),('G','H'),
             ('A','D'),('B','E'),('C','F'),('D','G'),('E','H'),
             ('B','F'),('D','E')]

    # Draw edges
    for n1, n2 in edges:
        x1, y1 = nodes[n1]; x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], color='gray', lw=1.2, alpha=0.5, zorder=1)

    # Draw nodes
    focus = 'E'
    for name, (x, y) in nodes.items():
        colour = UP_GOLD if name == focus else (
            UP_BLUE if name in [n for n1,n2 in edges
                                 for n in [n1,n2] if focus in [n1,n2]
                                 and n != focus] else 'steelblue')
        ax.add_patch(plt.Circle((x, y), 0.32, color=colour,
                                 alpha=0.9, zorder=5))
        ax.text(x, y, name, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=6)

    # Message arrows pointing to E
    neighbours = [n for n1,n2 in edges for n in [n1,n2]
                  if focus in [n1,n2] and n != focus]
    for nb in set(neighbours):
        x1, y1 = nodes[nb]; x2, y2 = nodes[focus]
        dx, dy = x2-x1, y2-y1
        norm = np.sqrt(dx**2+dy**2)
        ax.annotate('', xy=(x2-0.36*dx/norm, y2-0.36*dy/norm),
                    xytext=(x1+0.36*dx/norm, y1+0.36*dy/norm),
                    arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2.0))

    ax.text(5.0, 0.4, 'Node E aggregates messages from all neighbours',
            ha='center', fontsize=8.5, color='gray', style='italic')

    # Legend
    for colour, label in [(UP_GOLD, 'Focus node'), (UP_BLUE, 'Neighbours'),
                           ('steelblue', 'Other nodes'), (ACCENT, 'Messages')]:
        pass  # simplified — colours speak for themselves

    _method_box(axes[1], 'Graph Neural Networks (GNNs)',
                'Battaglia et al. (2018), Pfaff et al. (2021)',
                'teal', [
                    '• Represent CFD mesh as graph:',
                    '  nodes = mesh points, edges = connectivity',
                    '',
                    '• Message passing: each node aggregates',
                    '  information from its neighbours',
                    '',
                    r'  $h_i^{(l+1)} = \phi\!\left(h_i^{(l)},\; \bigoplus_{j \in \mathcal{N}(i)} \psi(h_i^{(l)}, h_j^{(l)})\right)$',
                    '',
                    '• Naturally handles unstructured meshes',
                    '  — no interpolation to regular grid needed',
                    '',
                    '• CFD use: MeshGraphNet (DeepMind),',
                    '  aerodynamics surrogate models',
                ])

    plt.suptitle('Graph Neural Networks — Learning on Meshes',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_gan():
    """GAN architecture schematic for flow field generation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]; ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('GAN — Adversarial Training',
                 fontsize=11, color=UP_BLUE, fontweight='bold')

    def rblock(ax, x, y, w, h, col, label, sub=''):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                      boxstyle='round,pad=0.15', color=col, alpha=0.85, zorder=3))
        ax.text(x+w/2, y+h/2+(0.15 if sub else 0), label,
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=4)
        if sub:
            ax.text(x+w/2, y+h/2-0.25, sub, ha='center', va='center',
                    fontsize=7.5, color='white', alpha=0.9, zorder=4)

    # Noise input
    rblock(ax, 0.2, 2.8, 1.4, 1.4, 'gray',    '$z$', 'random\nnoise')
    # Generator
    rblock(ax, 2.2, 2.5, 2.0, 2.0, UP_BLUE,   'Generator\n$G$', '$G(z)$')
    # Fake sample
    rblock(ax, 5.0, 2.5, 1.6, 2.0, 'steelblue','Fake\nfield', '$G(z)$')
    # Real sample
    rblock(ax, 5.0, 0.2, 1.6, 1.8, UP_GOLD,   'Real\nfield', 'DNS/exp')
    # Discriminator
    rblock(ax, 7.4, 1.8, 2.0, 2.4, ACCENT,    'Discriminator\n$D$', 'Real or fake?')
    # Output
    rblock(ax, 10.2, 2.5, 1.5, 1.0, 'gray',   '$D(x)$', '[0, 1]')

    # Arrows
    for src, dst in [
        ((1.6, 3.5), (2.2, 3.5)),
        ((4.2, 3.5), (5.0, 3.5)),
        ((6.6, 3.5), (7.4, 3.5)),
        ((6.6, 1.1), (7.4, 2.5)),
        ((9.4, 3.0), (10.2, 3.0)),
    ]:
        ax.annotate('', xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Feedback arrows
    ax.annotate('', xy=(4.2, 4.2), xytext=(10.2, 4.2),
                arrowprops=dict(arrowstyle='->', color=UP_BLUE, lw=1.5,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(7.0, 5.2, 'Generator loss:\nfool discriminator',
            ha='center', fontsize=8, color=UP_BLUE, style='italic')

    ax.annotate('', xy=(10.2, 1.8), xytext=(9.4, 1.8),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.0))
    ax.text(11.0, 1.5, 'D loss:\nclassify\ncorrectly',
            ha='center', fontsize=7.5, color=ACCENT, style='italic')

    _method_box(axes[1], 'Generative Adversarial Networks (GANs)',
                'Goodfellow et al. (2014)',
                ACCENT, [
                    '• Generator $G$: maps noise $z$ → synthetic field',
                    '• Discriminator $D$: real vs generated?',
                    '',
                    '• Adversarial training:',
                    '  $G$ tries to fool $D$',
                    '  $D$ tries to correctly classify',
                    '',
                    r'  $\min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$',
                    '',
                    '• CFD use: turbulence field synthesis,',
                    '  super-resolution of coarse LES,',
                    '  data augmentation',
                ])

    plt.suptitle('Generative Adversarial Networks — Learning Flow Distributions',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_rnn():
    """RNN / LSTM unrolled architecture schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]; ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('RNN — Unrolled Through Time',
                 fontsize=11, color=UP_BLUE, fontweight='bold')

    ts = [1.0, 3.8, 6.6, 9.4]
    labels_t = ['$t-2$', '$t-1$', '$t$', '$t+1$']

    for i, (x, tl) in enumerate(zip(ts, labels_t)):
        # Input
        ax.add_patch(mpatches.FancyBboxPatch((x-0.5, 0.5), 1.0, 0.9,
                      boxstyle='round,pad=0.1', color='steelblue',
                      alpha=0.8, zorder=3))
        ax.text(x, 0.95, f'$x_{{{tl[1:-1]}}}$', ha='center',
                fontsize=9, color='white', fontweight='bold', zorder=4)

        # Hidden state
        ax.add_patch(mpatches.FancyBboxPatch((x-0.6, 2.2), 1.2, 1.2,
                      boxstyle='round,pad=0.1', color=UP_BLUE,
                      alpha=0.9, zorder=3))
        ax.text(x, 2.8, f'$h_{{{tl[1:-1]}}}$', ha='center',
                fontsize=9, color='white', fontweight='bold', zorder=4)

        # Output
        if i < len(ts) - 1:
            ax.add_patch(mpatches.FancyBboxPatch((x-0.5, 4.2), 1.0, 0.9,
                          boxstyle='round,pad=0.1', color=UP_GOLD,
                          alpha=0.8, zorder=3))
            ax.text(x, 4.65, f'$y_{{{tl[1:-1]}}}$', ha='center',
                    fontsize=9, color='white', fontweight='bold', zorder=4)

        # Input → hidden
        ax.annotate('', xy=(x, 2.2), xytext=(x, 1.4),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
        # Hidden → output
        if i < len(ts) - 1:
            ax.annotate('', xy=(x, 4.2), xytext=(x, 3.4),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    # Recurrent connections h_t → h_{t+1}
    for i in range(len(ts) - 1):
        ax.annotate('', xy=(ts[i+1]-0.6, 2.8), xytext=(ts[i]+0.6, 2.8),
                    arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2.0))

    ax.text(5.2, 1.8, 'hidden state carries\ntemporal memory →',
            ha='center', fontsize=8, color=ACCENT, style='italic')

    # Equation
    ax.text(5.5, 5.8,
            r'$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$',
            ha='center', fontsize=10, color=UP_BLUE, fontweight='bold')

    _method_box(axes[1], 'Recurrent Neural Networks (RNNs / LSTMs)',
                'Hochreiter & Schmidhuber (1997) — LSTM',
                UP_BLUE, [
                    '• Hidden state $h_t$ carries memory across steps',
                    '',
                    r'  $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$',
                    '',
                    '• LSTM adds gates to control memory:',
                    '  forget gate, input gate, output gate',
                    '  → mitigates vanishing gradient over long sequences',
                    '',
                    '• CFD use:',
                    '  time series of sensor readings,',
                    '  reduced-order model time stepping,',
                    '  sequence-to-sequence flow prediction',
                ])

    plt.suptitle('Recurrent Neural Networks — Temporal Memory',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_reservoir_computing():
    """Echo State Network / Reservoir Computing schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]; ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('Echo State Network — Reservoir Computing',
                 fontsize=11, color=UP_BLUE, fontweight='bold')

    # Input layer
    ax.add_patch(mpatches.FancyBboxPatch((0.2, 2.5), 1.2, 2.0,
                  boxstyle='round,pad=0.1', color='steelblue',
                  alpha=0.8, zorder=3))
    ax.text(0.8, 3.5, 'Input\n$u(t)$', ha='center', fontsize=9,
            color='white', fontweight='bold', zorder=4)

    # Reservoir (random, fixed)
    ax.add_patch(plt.Circle((5.5, 3.5), 2.2, color='lightgray',
                              alpha=0.5, zorder=2))
    ax.add_patch(plt.Circle((5.5, 3.5), 2.2, fill=False,
                              edgecolor=UP_GOLD, lw=2.0, ls='--', zorder=3))
    ax.text(5.5, 5.9, 'Reservoir (fixed, random)',
            ha='center', fontsize=9, color=UP_GOLD, fontweight='bold')

    # Reservoir nodes
    rng_plot = np.random.default_rng(42)
    r_nodes = [(5.5 + 1.5*np.cos(a), 3.5 + 1.5*np.sin(a))
               for a in np.linspace(0, 2*np.pi, 10, endpoint=False)]
    r_nodes += [(5.5 + 0.7*np.cos(a), 3.5 + 0.7*np.sin(a))
                for a in np.linspace(0.3, 2*np.pi+0.3, 5, endpoint=False)]

    # Random internal connections
    for _ in range(18):
        i, j = rng_plot.integers(0, len(r_nodes), 2)
        if i != j:
            ax.plot([r_nodes[i][0], r_nodes[j][0]],
                    [r_nodes[i][1], r_nodes[j][1]],
                    color='gray', alpha=0.3, lw=0.8)
    for (rx, ry) in r_nodes:
        ax.add_patch(plt.Circle((rx, ry), 0.18, color=UP_BLUE,
                                 alpha=0.7, zorder=5))

    # Output layer (trained)
    ax.add_patch(mpatches.FancyBboxPatch((10.0, 2.5), 1.5, 2.0,
                  boxstyle='round,pad=0.1', color=ACCENT,
                  alpha=0.85, zorder=3))
    ax.text(10.75, 3.5, 'Output\n$y(t)$\n(trained)',
            ha='center', fontsize=9, color='white', fontweight='bold', zorder=4)

    # Arrows: input → reservoir
    ax.annotate('', xy=(3.3, 3.5), xytext=(1.4, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(2.35, 3.8, '$W_{in}$\n(fixed)', ha='center',
            fontsize=8, color='gray')

    # Reservoir → output
    ax.annotate('', xy=(10.0, 3.5), xytext=(7.7, 3.5),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2.0))
    ax.text(8.85, 3.9, '$W_{out}$\n(trained!)', ha='center',
            fontsize=8.5, color=ACCENT, fontweight='bold')

    # Feedback arrow
    ax.annotate('', xy=(5.5, 1.3), xytext=(10.75, 1.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                connectionstyle='arc3,rad=0'))
    ax.plot([10.75, 10.75], [2.5, 1.3], color='gray', lw=1.0)
    ax.plot([5.5, 5.5], [1.3, 1.2], color='gray', lw=1.0)
    ax.annotate('', xy=(5.5, 1.4), xytext=(5.5, 1.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax.text(8.0, 0.7, 'optional feedback', ha='center',
            fontsize=7.5, color='gray', style='italic')

    ax.text(5.5, 0.2, 'Only $W_{out}$ is trained — reservoir is fixed',
            ha='center', fontsize=8.5, color=UP_GOLD, fontweight='bold')

    _method_box(axes[1], 'Reservoir Computing (RC)',
                'Echo State Networks — Jaeger (2001)',
                UP_GOLD, [
                    '• Fixed random recurrent network = reservoir',
                    '• Only the output layer $W_{out}$ is trained',
                    '  → linear regression, not backprop',
                    '',
                    '• Reservoir projects input into high-dimensional',
                    '  nonlinear space — rich feature representation',
                    '',
                    '• Physics interpretation: reservoir is a fixed',
                    '  dynamical system acting as a nonlinear filter',
                    '',
                    '• CFD use: chaotic system prediction,',
                    '  reduced-order modelling of attractors,',
                    '  very fast training vs deep RNNs',
                ])

    plt.suptitle('Reservoir Computing — Fixed Dynamics, Trained Readout',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_foundation_models():
    """Foundation model / Transformer architecture schematic for science."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('Transformer — Self-Attention Block',
                 fontsize=11, color=UP_BLUE, fontweight='bold')

    def tblock(ax, x, y, w, h, col, label):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                      boxstyle='round,pad=0.1', color=col, alpha=0.85, zorder=3))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', zorder=4)

    # Input tokens
    for i, label in enumerate(['$x_1$', '$x_2$', '$x_3$', '$\cdots$']):
        tblock(ax, 0.3+i*1.8, 0.4, 1.3, 0.8, 'steelblue', label)

    # Multi-head attention
    tblock(ax, 0.5, 2.0, 8.5, 1.4, UP_BLUE,
           'Multi-Head Self-Attention\n$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^T/\sqrt{d_k})V$')

    # Add & Norm
    tblock(ax, 0.5, 3.8, 8.5, 0.8, 'teal', 'Add & Layer Norm')

    # FFN
    tblock(ax, 0.5, 5.0, 8.5, 1.2, 'steelblue',
           'Feed-Forward Network\n$\mathrm{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$')

    # Add & Norm
    tblock(ax, 0.5, 6.6, 8.5, 0.8, 'teal', 'Add & Layer Norm')

    # Arrows
    ax.annotate('', xy=(4.75, 2.0), xytext=(4.75, 1.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(4.75, 3.8), xytext=(4.75, 3.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(4.75, 5.0), xytext=(4.75, 4.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(4.75, 6.6), xytext=(4.75, 6.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Residual connection
    ax.plot([0.2, 0.2], [1.2, 7.5], color='gray', lw=1.0, ls='--', alpha=0.6)
    ax.annotate('', xy=(0.5, 7.1), xytext=(0.2, 7.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax.text(0.08, 4.5, 'Residual\nconnection',
            ha='center', fontsize=7, color='gray',
            style='italic', rotation=90)

    _method_box(axes[1], 'Foundation Models for Science',
                'Aurora, Poseidon, ClimaX, Neural GCM',
                ACCENT, [
                    '• Transformer architecture trained at massive scale',
                    '  on simulation and observational datasets',
                    '',
                    '• Self-attention: every token attends to every other',
                    '  → global context in one operation',
                    '',
                    '• Pre-trained once, fine-tuned per task',
                    '  (analogous to GPT for language)',
                    '',
                    '• Examples:',
                    '  Aurora (Microsoft) — global weather 10-day forecast',
                    '  Poseidon — ocean circulation',
                    '  Neural GCM (Google DeepMind) — atmosphere model',
                    '  ClimaX — climate projection',
                    '',
                    '• CFD status: active research — not yet standard',
                ])

    plt.suptitle('Foundation Models — Pre-trained at Scale, Fine-tuned per Task',
                 fontsize=12, color=UP_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION — Von Kármán vortex street time evolution
# ═══════════════════════════════════════════════════════════════════════════════

def animate_cylinder_wake(model, data_path, norm_path, device='cpu',
                           n_frames=40, field='speed', figsize=(13, 4)):
    """
    Animate the cylinder wake fields predicted by the pre-trained NS PINN.

    Parameters
    ----------
    model     : NS_PINN instance (pre-trained, eval mode)
    data_path : str — path to cylinder_nektar_wake.mat
    norm_path : str — path to ns_pinn_normalisation.npz (saved by training script)
    device    : str
    n_frames  : int — number of animation frames
    field     : 'speed' | 'u' | 'v' | 'vorticity'

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
           Display with: IPython.display.HTML(anim.to_jshtml())
    """
    import scipy.io
    import torch
    import matplotlib.animation as animation

    # ── Load dataset and normalisation constants ───────────────────────────────
    data   = scipy.io.loadmat(data_path)
    X_star = data['X_star']
    t_star = data['t'].flatten()

    norm       = np.load(norm_path)
    x_min, x_max = float(norm['x_min']), float(norm['x_max'])
    y_min, y_max = float(norm['y_min']), float(norm['y_max'])
    t_min, t_max = float(norm['t_min']), float(norm['t_max'])
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    t_scale = t_max - t_min

    # ── Evaluation grid — stay within training domain ─────────────────────────
    # x_arr = np.linspace(x_min, x_max, 200)
    # y_arr = np.linspace(y_min, y_max, 80)
    # Xg, Yg = np.meshgrid(x_arr, y_arr)
    # N_grid  = Xg.size

    # Normalise grid to [0,1] — must match training normalisation exactly
    # Xg_n = (Xg - x_min) / x_scale
    # Yg_n = (Yg - y_min) / y_scale

    # Extend grid upstream to show cylinder — model extrapolates cleanly here
    # since upstream flow is nearly uniform (u≈1, v≈0)
    x_arr = np.linspace(-1.0, x_max, 220)   # hard-code -1.0 as upstream limit
    y_arr = np.linspace(y_min, y_max, 80)
    Xg, Yg = np.meshgrid(x_arr, y_arr)
    N_grid  = Xg.size   # add this line

    # Normalise using training constants — values outside [0,1] are extrapolation
    Xg_n = (Xg - x_min) / x_scale
    Yg_n = (Yg - y_min) / y_scale

    t_frames = np.linspace(t_min, t_max, n_frames)

    def to_t(a):
        return torch.tensor(a.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

    # ── Pre-compute all frames ────────────────────────────────────────────────
    print(f"Pre-computing {n_frames} animation frames...")
    frames_data = []
    model.eval()

    for i, t_eval in enumerate(t_frames):
        t_n = (t_eval - t_min) / t_scale   # normalise time
        with torch.no_grad():
            u_p, v_p, _ = model(to_t(Xg_n), to_t(Yg_n),
                                  torch.full((N_grid, 1), t_n,
                                             dtype=torch.float32).to(device))
        u_f = u_p.cpu().numpy().reshape(Xg.shape)
        v_f = v_p.cpu().numpy().reshape(Xg.shape)

        if field == 'speed':
            F = np.sqrt(u_f**2 + v_f**2)
        elif field == 'u':
            F = u_f
        elif field == 'v':
            F = v_f
        elif field == 'vorticity':
            dv_dx = np.gradient(v_f, x_arr, axis=1)
            du_dy = np.gradient(u_f, y_arr, axis=0)
            F = dv_dx - du_dy
        else:
            F = np.sqrt(u_f**2 + v_f**2)

        frames_data.append(F)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_frames} frames computed")

    print("All frames ready — building animation...")

    field_labels = {'speed': 'Speed [m/s]', 'u': 'u-velocity [m/s]',
                    'v': 'v-velocity [m/s]', 'vorticity': 'Vorticity [1/s]'}
    cmaps = {'speed': 'hot', 'u': 'RdBu_r', 'v': 'RdBu_r', 'vorticity': 'RdBu_r'}

    F_all = np.stack(frames_data)
    vmin  = np.percentile(F_all, 2)
    vmax  = np.percentile(F_all, 98)

    theta = np.linspace(0, 2*np.pi, 100)
    cyl_r = 0.5   # cylinder radius

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame):
        ax.clear()
        ax.contourf(Xg, Yg, frames_data[frame], 40,
                     cmap=cmaps.get(field, 'hot'), vmin=vmin, vmax=vmax)
        ax.fill(cyl_r * np.cos(theta), cyl_r * np.sin(theta),
                color='gray', zorder=5)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(
            f'NS PINN — {field_labels.get(field, field)} | '
            f't = {t_frames[frame]:.2f} s',
            fontsize=10, color=UP_BLUE, fontweight='bold')
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=120, blit=False)

    plt.close(fig)
    print("Animation ready. Display with: HTML(anim.to_jshtml())")
    return anim


def animate_vortex_synthetic(n_frames=40, field='speed', figsize=(13, 4)):
    """
    Animate a synthetic Von Kármán vortex street using superposed point vortices.
    Used as fallback when the pre-trained model or dataset is unavailable.

    Parameters
    ----------
    n_frames : int — number of animation frames
    field    : 'speed' | 'vorticity'

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    import matplotlib.animation as animation

    x_arr = np.linspace(-2.5, 9, 220)
    y_arr = np.linspace(-3.0, 3.0, 90)
    Xg, Yg = np.meshgrid(x_arr, y_arr)
    omega = 2 * np.pi * 0.17  # Strouhal shedding frequency at Re=100
    t_frames = np.linspace(0, 2*np.pi/omega, n_frames)

    def compute_field(t):
        U = np.ones_like(Xg)
        V = np.zeros_like(Xg)
        vortex_xs = [1.5 + i * 1.2 for i in range(6)]
        for i, vx in enumerate(vortex_xs):
            sign   =  1 if i % 2 == 0 else -1
            vy_u   =  0.5 * np.cos(omega * t + i * np.pi)
            vy_l   = -0.5 * np.cos(omega * t + i * np.pi)
            s      = 0.35
            r2_u   = (Xg - vx)**2 + (Yg - vy_u)**2 + 0.01
            r2_l   = (Xg - vx)**2 + (Yg - vy_l)**2 + 0.01
            U += -sign * s * (Yg - vy_u) / r2_u
            V +=  sign * s * (Xg - vx)   / r2_u
            U +=  sign * s * (Yg - vy_l) / r2_l
            V += -sign * s * (Xg - vx)   / r2_l
        mask = Xg**2 + Yg**2 < 0.26
        U[mask] = np.nan; V[mask] = np.nan
        if field == 'vorticity':
            dv = np.gradient(V, x_arr, axis=1)
            du = np.gradient(U, y_arr, axis=0)
            return dv - du
        return np.sqrt(U**2 + V**2)

    # Pre-compute
    frames_data = [compute_field(t) for t in t_frames]
    F_all = np.stack([f for f in frames_data if not np.all(np.isnan(f))])
    vmin = np.nanpercentile(F_all, 2)
    vmax = np.nanpercentile(F_all, 98)

    cmap  = 'RdBu_r' if field == 'vorticity' else 'hot'
    label = 'Vorticity [1/s]' if field == 'vorticity' else 'Speed [m/s]'
    theta = np.linspace(0, 2*np.pi, 100)

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame):
        ax.clear()
        ax.contourf(Xg, Yg, frames_data[frame], 40,
                     cmap=cmap, vmin=vmin, vmax=vmax)
        ax.fill(0.5*np.cos(theta), 0.5*np.sin(theta), color='gray', zorder=5)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=11); ax.set_ylabel('y', fontsize=11)
        ax.set_title(
            f'Von Kármán Vortex Street — {label} | '
            f't = {t_frames[frame]:.2f} s  (synthetic)',
            fontsize=10, color=UP_BLUE, fontweight='bold')
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=120, blit=False)
    plt.close(fig)
    return anim
