"""
Mock data for PathSight AI well planning application.

Each surface point (x, y) represents a previously drilled well location.
Each location has one or more geological units with depth intervals,
and each unit has a distribution (mean, variance) of reservoir thickness.
"""

import numpy as np

# ── Mock well locations at surface (x, y, z=0) ──────────────────────────
# Each entry: { "name": str, "x": float, "y": float, "units": [...] }
# "units" is a list of geological units encountered at that location,
# each with: { "name": str, "depth_from": float, "depth_to": float,
#               "thickness_mean": float, "thickness_var": float }

WELL_LOCATIONS = [
    {
        "name": "Well-A",
        "x": 500.0,
        "y": 600.0,
        "units": [
            {"name": "2E", "depth_from": 0,    "depth_to": 800,
             "thickness_mean": 12.5, "thickness_var": 2.1},
            {"name": "2D", "depth_from": 800,   "depth_to": 1600,
             "thickness_mean": 8.3,  "thickness_var": 1.4},
            {"name": "2C", "depth_from": 1600,  "depth_to": 2400,
             "thickness_mean": 15.0, "thickness_var": 3.0},
        ],
    },
    {
        "name": "Well-B",
        "x": 1200.0,
        "y": 400.0,
        "units": [
            {"name": "2E", "depth_from": 0,    "depth_to": 750,
             "thickness_mean": 11.0, "thickness_var": 1.8},
            {"name": "2D", "depth_from": 750,   "depth_to": 1500,
             "thickness_mean": 9.1,  "thickness_var": 2.0},
        ],
    },
    {
        "name": "Well-C",
        "x": 800.0,
        "y": 1100.0,
        "units": [
            {"name": "2E", "depth_from": 0,    "depth_to": 900,
             "thickness_mean": 13.2, "thickness_var": 2.5},
            {"name": "2D", "depth_from": 900,   "depth_to": 1700,
             "thickness_mean": 7.8,  "thickness_var": 1.2},
            {"name": "2C", "depth_from": 1700,  "depth_to": 2600,
             "thickness_mean": 14.5, "thickness_var": 2.8},
        ],
    },
    {
        "name": "Well-D",
        "x": 1500.0,
        "y": 1300.0,
        "units": [
            {"name": "2E", "depth_from": 0,    "depth_to": 850,
             "thickness_mean": 10.5, "thickness_var": 1.6},
            {"name": "2D", "depth_from": 850,   "depth_to": 1650,
             "thickness_mean": 8.9,  "thickness_var": 1.9},
        ],
    },
    {
        "name": "Well-E",
        "x": 300.0,
        "y": 1400.0,
        "units": [
            {"name": "2E", "depth_from": 0,    "depth_to": 700,
             "thickness_mean": 14.0, "thickness_var": 3.2},
            {"name": "2D", "depth_from": 700,   "depth_to": 1400,
             "thickness_mean": 6.5,  "thickness_var": 1.0},
            {"name": "2C", "depth_from": 1400,  "depth_to": 2200,
             "thickness_mean": 16.2, "thickness_var": 3.5},
        ],
    },
    {
        "name": "Well-F",
        "x": 1800.0,
        "y": 800.0,
        "units": [
            {"name": "2E", "depth_from": 0,    "depth_to": 820,
             "thickness_mean": 11.8, "thickness_var": 2.0},
            {"name": "2D", "depth_from": 820,   "depth_to": 1620,
             "thickness_mean": 9.5,  "thickness_var": 1.7},
        ],
    },
]


def generate_vertical_well_path(x: float, y: float,
                                 max_depth: float = 3000.0,
                                 step: float = 1.0) -> np.ndarray:
    """
    Generate a straight-line (vertical) well path from surface to max_depth.

    Returns an (N, 3) array where each row is (x, y, z).
    z goes from 0 (surface) to max_depth.
    """
    z = np.arange(0, max_depth + step, step)
    xs = np.full_like(z, x)
    ys = np.full_like(z, y)
    return np.column_stack([xs, ys, z])
