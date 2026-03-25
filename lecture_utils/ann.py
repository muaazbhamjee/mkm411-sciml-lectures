"""
lecture_utils/ann.py
====================
Demo neural network classes for Lecture 1.

SmallNet  — tiny 2-input network for forward pass demonstration
DemoANN   — configurable fully-connected network for loss landscape demo
"""

import torch
import torch.nn as nn
from .config import SEED


class SmallNet(nn.Module):
    """
    Tiny 3-layer network for forward pass demonstration.

    Architecture: 2 → 8 → 8 → 1  (tanh activations)
    Fixed seed for reproducible demonstration.

    Input : (x̂, t̂) — two normalised inputs for clarity
    Output: scalar prediction

    Note: The project PINN uses 6 inputs (x, y, t, rho, cp, k).
    This simplified version is used in lectures to illustrate the
    forward pass without overwhelming the diagram.
    """

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

    def forward(self, x):
        return self.net(x)

    def forward_with_activations(self, x):
        """
        Forward pass returning intermediate activations at each layer.

        Used by the forward_pass_widget to visualise signal propagation.

        Parameters
        ----------
        x : Tensor (1, 2) — normalised inputs [x̂, t̂]

        Returns
        -------
        output      : Tensor (1, 1) — final prediction
        activations : list of ndarray — one per layer including input/output
        """
        activations = [x.detach().cpu().numpy()]
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Tanh):
                activations.append(h.detach().cpu().numpy())
        activations.append(h.detach().cpu().numpy())
        return h, activations


class DemoANN(nn.Module):
    """
    Configurable fully-connected ANN for loss landscape demonstration.

    Learns a simple 1D regression task (sin(x)) to illustrate the
    loss landscape and gradient descent.

    Architecture: 1 → n_hidden × n_neurons → 1  (tanh activations)
    """

    def __init__(self, n_hidden=2, n_neurons=4):
        super().__init__()
        torch.manual_seed(SEED)
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

    def get_first_weight(self):
        """Return the first weight used in loss landscape exploration."""
        return self.net[0].weight.data[0, 0].item()

    def set_first_weight(self, value):
        """Set the first weight for loss landscape exploration."""
        self.net[0].weight.data[0, 0] = value

    def compute_loss(self, X, y):
        """MSE loss for the demo regression task."""
        import torch.nn as nn
        return nn.MSELoss()(self.forward(X), y)
