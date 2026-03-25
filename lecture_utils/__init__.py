"""
lecture_utils/
==============
Utility package for MKM411 lecture notebooks.

Modules
-------
config   — shared constants and colour palette
ann      — SmallNet and DemoANN for Lecture 1 demonstrations
pinn     — MinimalHeatPINN and NS_PINN for Lecture 2
plots    — all plotting functions (theory figures + panorama figures)
widgets  — three interactive ipywidgets for Lecture 1

Training script
---------------
lecture_utils/train_ns_pinn.py — run overnight on Mjolnir
"""

from .config  import UP_BLUE, UP_GOLD, ACCENT, SEED, FIGSIZE_WIDE
from .ann     import SmallNet, DemoANN
from .pinn    import MinimalHeatPINN, NS_PINN
from .plots   import (
    # Architecture / theory figures
    draw_network,
    plot_feedforward,
    plot_xavier_initialisation,
    plot_backpropagation,
    plot_bias_variance,
    # Landscape and comparison
    plot_sciml_landscape,
    plot_method_comparison,
    plot_loss_comparison,
    # Cylinder wake
    plot_cylinder_fields,
    plot_vortex_street,
    # Panorama — one per method
    plot_deeponet,
    plot_fno,
    plot_gnn,
    plot_gan,
    plot_rnn,
    plot_reservoir_computing,
    plot_foundation_models,
)
from .widgets import (
    activation_explorer,
    forward_pass_widget,
    loss_landscape_widget,
)
