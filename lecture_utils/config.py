"""
lecture_utils/config.py
=======================
Shared constants and styling for MKM411 lecture notebooks.
"""

import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Plot styling ──────────────────────────────────────────────────────────────
UP_BLUE  = "#002B5C"
UP_GOLD  = "#B48C3C"
ACCENT   = "#E63946"

FIGSIZE_WIDE   = (13, 4)
FIGSIZE_SQUARE = (6, 5)
FIGSIZE_TALL   = (8, 6)

# ── ANN demo parameters ───────────────────────────────────────────────────────
DEMO_ARCHITECTURE = [6, 40, 40, 40, 40, 40, 1]   # matches project model
