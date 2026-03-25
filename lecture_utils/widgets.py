"""
lecture_utils/widgets.py
========================
Three interactive widgets for Lecture 1:

1. activation_explorer  — plot activation functions and their derivatives
2. forward_pass_widget  — live forward pass through a small demo network
3. loss_landscape_widget — 1D loss landscape as a single weight varies

All widgets use ipywidgets and matplotlib.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import ipywidgets as widgets
from IPython.display import display, Image

from .config import UP_BLUE, UP_GOLD, ACCENT, SEED


def _fig_to_image(fig):
    """
    Render a matplotlib figure to an ipywidgets Image widget via PNG buffer.
    Completely bypasses matplotlib's display hooks — no automatic rendering.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return widgets.Image(value=buf.read(), format='png')


# ── 1. Activation Function Explorer ──────────────────────────────────────────

def activation_explorer():
    """
    Interactive plot of activation functions and their first derivatives.

    Students can select which functions to display and see why tanh is
    preferred for PINNs (smooth, non-zero gradient everywhere, bounded).
    """
    x = np.linspace(-4, 4, 400)

    functions = {
        "tanh":    (np.tanh(x),
                    1 - np.tanh(x)**2,
                    UP_BLUE,   "tanh — smooth, bounded, non-zero gradient everywhere"),
        "ReLU":    (np.maximum(0, x),
                    (x > 0).astype(float),
                    ACCENT,    "ReLU — zero gradient for x<0, zero second derivative"),
        "Sigmoid": (1 / (1 + np.exp(-x)),
                    (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x))),
                    UP_GOLD,   "Sigmoid — saturates at extremes (vanishing gradient)"),
        "ELU":     (np.where(x >= 0, x, np.exp(x) - 1),
                    np.where(x >= 0, 1.0, np.exp(x)),
                    "green",   "ELU — smooth for x<0, linear for x>0"),
    }

    checks = {name: widgets.Checkbox(value=(name in ["tanh", "ReLU"]),
                                      description=name, indent=False)
              for name in functions}

    show_deriv = widgets.Checkbox(value=True,
                                   description="Show first derivative",
                                   indent=False)

    pinn_note = widgets.Checkbox(value=False,
                                  description="Highlight PINN requirement",
                                  indent=False)

    controls = widgets.VBox([
        widgets.HTML("<b>Select activation functions:</b>"),
        widgets.HBox(list(checks.values())),
        widgets.HBox([show_deriv, pinn_note]),
    ])

    out = widgets.Output()

    def update(*args):
        with plt.ioff():
            ncols = 2 if show_deriv.value else 1
            fig, axes = plt.subplots(1, ncols, figsize=(13, 4))
            if ncols == 1:
                axes = [axes]

            for name, (fx, dfx, colour, label) in functions.items():
                if checks[name].value:
                    axes[0].plot(x, fx, color=colour, linewidth=2.0, label=name)
                    if show_deriv.value:
                        axes[1].plot(x, dfx, color=colour, linewidth=2.0,
                                     linestyle="--", label=f"d/dx {name}")

            axes[0].set_title("Activation Functions $f(x)$",
                               fontsize=12, color=UP_BLUE)
            axes[0].axhline(0, color="gray", linewidth=0.5)
            axes[0].axvline(0, color="gray", linewidth=0.5)
            axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$f(x)$")
            axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(-1.5, 1.5)

            if show_deriv.value:
                axes[1].set_title("First Derivative $f'(x)$",
                                   fontsize=12, color=UP_BLUE)
                axes[1].axhline(0, color="gray", linewidth=0.5)
                axes[1].axvline(0, color="gray", linewidth=0.5)
                axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$f'(x)$")
                axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim(-0.3, 1.3)

                if pinn_note.value:
                    axes[1].axhline(0, color=ACCENT, linewidth=1.5,
                                    linestyle=":", alpha=0.8)
                    axes[1].text(3.5, 0.05,
                                 "PINNs need\n$f'(x) \\neq 0$",
                                 color=ACCENT, fontsize=8, ha="right")

            plt.suptitle(
                "Key insight: PINNs compute $\\partial^2 T/\\partial x^2$ "
                "via autograd — the activation must be twice differentiable "
                "with non-zero derivatives",
                fontsize=10, color="gray", style="italic"
            )
            plt.tight_layout()
            img = _fig_to_image(fig)

        with out:
            out.clear_output(wait=True)
            display(img)

    for cb in list(checks.values()) + [show_deriv, pinn_note]:
        cb.observe(update, names="value")

    update()
    display(controls, out)


# ── 2. Forward Pass Widget ────────────────────────────────────────────────────

class _SmallNet(nn.Module):
    """Tiny 3-layer network for forward pass demonstration."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.Tanh(),
            nn.Linear(8, 8), nn.Tanh(),
            nn.Linear(8, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward_with_activations(self, x):
        """Return output and intermediate activations for each layer."""
        activations = [x.detach().numpy()]
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Tanh):
                activations.append(h.detach().numpy())
        activations.append(h.detach().numpy())
        return h, activations

_demo_net = _SmallNet()

def forward_pass_widget():
    """
    Interactive forward pass visualiser.

    Students control two inputs (x, t) and see how the signal propagates
    through the network layer by layer, with activation values shown.
    Builds intuition for what a forward pass actually computes.
    """
    x_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01,
                                    description="$\\hat{x}$:", continuous_update=True,
                                    style={"description_width": "40px"})
    t_slider = widgets.FloatSlider(value=0.3, min=0.0, max=1.0, step=0.01,
                                    description="$\\hat{t}$:", continuous_update=True,
                                    style={"description_width": "40px"})

    note = widgets.HTML(
        "<small><i>This demo uses a 2-input network (x, t) for clarity. "
        "The project PINN uses 6 inputs: (x, y, t, ρ, c_p, k).</i></small>"
    )

    out = widgets.Output()

    def update(*args):
        with plt.ioff():
            x_val = x_slider.value
            t_val = t_slider.value
            inp = torch.tensor([[x_val, t_val]], dtype=torch.float32)
            _, activations = _demo_net.forward_with_activations(inp)

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            layer_names = ["Input\n[x̂, t̂]",
                           "Layer 1\n(8 neurons)",
                           "Layer 2\n(8 neurons)",
                           "Output\nT [K]"]
            colours = [UP_BLUE, UP_BLUE, UP_BLUE, UP_GOLD]

            for ax, act, name, col in zip(axes, activations, layer_names, colours):
                vals = act.flatten()
                if len(vals) == 1:
                    ax.barh([0], vals, color=col, alpha=0.8)
                    ax.set_yticks([0])
                    ax.set_yticklabels(["T"])
                    ax.set_xlim(-2, 2)
                    ax.set_title(f"{name}\n{vals[0]:.4f}", fontsize=10,
                                  color=UP_BLUE)
                else:
                    ax.barh(range(len(vals)), vals, color=col, alpha=0.7)
                    ax.set_yticks(range(len(vals)))
                    ax.set_yticklabels([f"n{i+1}" for i in range(len(vals))],
                                        fontsize=7)
                    ax.set_xlim(-1.5, 1.5)
                    ax.set_title(name, fontsize=10, color=UP_BLUE)
                ax.axvline(0, color="gray", linewidth=0.5)
                ax.grid(True, alpha=0.2, axis="x")

            plt.suptitle(
                f"Forward pass: $\\hat{{x}}$ = {x_val:.2f},  "
                f"$\\hat{{t}}$ = {t_val:.2f}  →  "
                f"T = {activations[-1].flatten()[0]:.4f}",
                fontsize=11, color=UP_BLUE, fontweight="bold"
            )
            plt.tight_layout()
            img = _fig_to_image(fig)

        with out:
            out.clear_output(wait=True)
            display(img)

    x_slider.observe(update, names="value")
    t_slider.observe(update, names="value")
    update()
    display(widgets.VBox([
        widgets.HBox([x_slider, t_slider]),
        note,
        out
    ]))


# ── 3. Loss Landscape Widget ──────────────────────────────────────────────────

def loss_landscape_widget():
    """
    1D loss landscape visualiser.

    Fixes all weights except one and plots the MSE loss as that weight varies.
    Illustrates: local minima, gradient descent steps, effect of learning rate.
    Students see why optimisation is non-trivial for neural networks.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Simple 1D regression task: learn sin(x)
    X_data = torch.linspace(0, 2*np.pi, 30).unsqueeze(1)
    y_data = torch.sin(X_data)

    # Tiny net: 1 → 4 → 1
    net = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    # We'll vary one specific weight and plot loss
    def compute_loss_for_weight(w_val, param_idx=0):
        """Temporarily set one weight and compute MSE."""
        original = net[0].weight.data[0, 0].item()
        net[0].weight.data[0, 0] = w_val
        with torch.no_grad():
            pred = net(X_data)
            loss = nn.MSELoss()(pred, y_data).item()
        net[0].weight.data[0, 0] = original
        return loss

    w_range = np.linspace(-3, 3, 300)
    loss_vals = np.array([compute_loss_for_weight(w) for w in w_range])

    current_w = widgets.FloatSlider(
        value=net[0].weight.data[0, 0].item(),
        min=-3.0, max=3.0, step=0.05,
        description="Weight $w$:", continuous_update=True,
        style={"description_width": "80px"}, layout={"width": "500px"}
    )

    lr_slider = widgets.FloatLogSlider(
        value=0.1, base=10, min=-2, max=0, step=0.1,
        description="Learning rate:", continuous_update=False,
        style={"description_width": "100px"}, layout={"width": "400px"}
    )

    step_btn = widgets.Button(description="Take one gradient step",
                               button_style="primary")
    reset_btn = widgets.Button(description="Reset", button_style="warning")

    out = widgets.Output()

    def plot(w_current):
        with plt.ioff():
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            axes[0].plot(w_range, loss_vals, color=UP_BLUE, linewidth=2)
            loss_at_w = compute_loss_for_weight(w_current)

            dw = 0.01
            grad_approx = (compute_loss_for_weight(w_current + dw) -
                           compute_loss_for_weight(w_current - dw)) / (2 * dw)

            axes[0].scatter([w_current], [loss_at_w],
                             color=ACCENT, s=120, zorder=5,
                             label=f"Current: w={w_current:.2f}, L={loss_at_w:.4f}")

            arrow_len = -lr_slider.value * grad_approx
            axes[0].annotate("",
                xy=(w_current + arrow_len * 0.8,
                    loss_at_w - abs(arrow_len) * 0.1),
                xytext=(w_current, loss_at_w),
                arrowprops=dict(arrowstyle="->", color=UP_GOLD, lw=2))

            axes[0].set_xlabel("Weight value $w$", fontsize=11)
            axes[0].set_ylabel("MSE Loss", fontsize=11)
            axes[0].set_title("Loss Landscape (one weight varied)",
                               fontsize=11, color=UP_BLUE)
            axes[0].legend(fontsize=9)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, min(loss_vals.max(), 5))

            net[0].weight.data[0, 0] = w_current
            with torch.no_grad():
                y_pred = net(X_data).numpy()

            axes[1].plot(X_data.numpy(), y_data.numpy(),
                         color=UP_BLUE, linewidth=2, label="Target: sin(x)")
            axes[1].plot(X_data.numpy(), y_pred,
                         color=ACCENT, linewidth=2,
                         linestyle="--", label="Network prediction")
            axes[1].set_xlabel("$x$", fontsize=11)
            axes[1].set_ylabel("Output", fontsize=11)
            axes[1].set_title("Network Prediction vs Target",
                               fontsize=11, color=UP_BLUE)
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)

            plt.suptitle(
                f"Gradient ≈ {grad_approx:.3f}  →  "
                f"Step = −lr × grad = −{lr_slider.value:.2f} × {grad_approx:.3f} "
                f"= {-lr_slider.value * grad_approx:.4f}",
                fontsize=10, color="gray", style="italic"
            )
            plt.tight_layout()
            img = _fig_to_image(fig)

        with out:
            out.clear_output(wait=True)
            display(img)

    def on_step(_):
        w_c = current_w.value
        dw = 0.01
        grad = (compute_loss_for_weight(w_c + dw) -
                compute_loss_for_weight(w_c - dw)) / (2 * dw)
        new_w = np.clip(w_c - lr_slider.value * grad, -3.0, 3.0)
        current_w.value = round(float(new_w), 3)

    def on_reset(_):
        current_w.value = round(float(
            torch.nn.init.xavier_normal_(
                torch.zeros(1, 1)).item()), 3)

    current_w.observe(lambda _: plot(current_w.value), names="value")
    step_btn.on_click(on_step)
    reset_btn.on_click(on_reset)

    plot(current_w.value)
    display(widgets.VBox([
        current_w, lr_slider,
        widgets.HBox([step_btn, reset_btn]),
        out
    ]))