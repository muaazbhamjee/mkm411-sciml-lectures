"""
lecture_utils/
==============
Utility package for MKM411 lecture notebooks.

Modules
-------
config   — shared constants and colour palette
ann      — SmallNet and DemoANN for Lecture 1 demonstrations
pinn     — MinimalHeatPINN and NS_PINN for Lecture 2
plots    — shared plotting functions (network diagram, landscape, cylinder)
widgets  — three interactive ipywidgets for Lecture 1

Training script
---------------
lecture_utils/train_ns_pinn.py — run overnight on Mjolnir to generate
                                  data/ns_pinn_pretrained.pt
"""

from .config  import UP_BLUE, UP_GOLD, ACCENT, SEED, FIGSIZE_WIDE
from .ann     import SmallNet, DemoANN
from .pinn    import MinimalHeatPINN, NS_PINN
from .plots   import (draw_network, plot_sciml_landscape,
                       plot_method_comparison, plot_loss_comparison,
                       plot_cylinder_fields, plot_vortex_street)
from .widgets import activation_explorer, forward_pass_widget, loss_landscape_widget
