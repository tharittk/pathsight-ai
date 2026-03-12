#!/usr/bin/env python3
"""
PathSight AI – Well Planning Optimization GUI

Layout
──────
┌───────────────────────────────┬──────────────────────┐
│                               │  Wellhead Location    │
│   3-D Visualisation           │  X: [____]  Y: [____] │
│   (matplotlib Axes3D)         │  [Set Wellhead]       │
│                               │                      │
│   • fault surfaces (3 bands)  │  ── Selected Wells ── │
│   • clickable well markers    │  (tree / list view)   │
│   • wellhead marker           │                      │
│   • optimised well path       │  [Run Optimisation]   │
└───────────────────────────────┴──────────────────────┘

On startup the app
  1. Initialises a RegularGrid3D spanning the G2-TKN fault area.
  2. Loads and processes Fault_2025_G2_TKN_01_W via preprocess_fault().
  3. Integrates the three output arrays (fault_plane, dip_side,
     anti_dip_side) into the grid.
  4. Displays all three fault point-clouds in the 3-D view.
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from mock_data import WELL_LOCATIONS, generate_vertical_well_path
from pathsight3d.container.grid import RegularGrid3D
from pathsight3d.fault.processing import preprocess_fault

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

# ── Fault file ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
FAULT_FILE = os.path.join(_HERE, "input", "faults", "Fault_2025_G2_TKN_01_W")

# ── Grid bounds (expanded ~1 000 m around the fault extent) ────────────
#   Fault X: ~911 000 – 918 700 m  →  grid: 910 000 – 920 000 m
#   Fault Y: ~827 000 – 840 500 m  →  grid: 825 000 – 842 000 m
#   Fault Z:  ~0 – 4 200 m depth   →  grid:       0 –   4 500 m
GRID_CFG = dict(
    x_min=910_000, x_max=920_000,
    y_min=825_000, y_max=842_000,
    z_min=0,       z_max=4_500,
    dx=100, dy=100, dz=100,
)

# ── max display points per fault cloud (subsampled for render speed) ────
_MAX_DISP = 30_000

# ── Colour palette ──────────────────────────────────────────────────────
CLR_WELL_DEFAULT   = "#1f77b4"   # blue  – unselected wells
CLR_WELL_SELECTED  = "#ff7f0e"   # orange – selected wells
CLR_WELLHEAD       = "#d62728"   # red   – wellhead platform
CLR_WELL_PATH      = "#2ca02c"   # green – optimised well path
CLR_FAULT_PLANE    = "#7f7f7f"   # grey  – interpolated fault surface
CLR_DIP_SIDE       = "#9467bd"   # purple – dip-side reservoir band
CLR_ANTI_DIP_SIDE  = "#17becf"   # cyan  – anti-dip-side reservoir band
MARKER_SIZE        = 80


class PathSightApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("PathSight AI – Well Planning Optimisation")
        self.geometry("1400x850")
        self.minsize(1100, 700)

        # ── State ───────────────────────────────────────────────────────
        self.wellhead_xy: tuple[float, float] | None = None
        self.selected_wells: dict[str, dict] = {}   # name → well dict
        self.well_path: np.ndarray | None = None

        # Fault / grid state (populated by _load_fault_and_grid)
        self.grid: RegularGrid3D | None = None
        # self.fault_plane: np.ndarray | None = None
        # self.dip_side: np.ndarray | None = None
        # self.anti_dip_side: np.ndarray | None = None
        # self._fault_error: str | None = None   # non-None if loading failed

        # ── Load fault + initialise grid (before building the UI so the
        #    status bar can show the outcome immediately) ─────────────────
        self._load_fault_and_grid()

        # ── UI construction ─────────────────────────────────────────────
        self._build_ui()
        self._draw_wells()

    # ────────────────────────────────────────────────────────────────────
    # Fault loading & grid initialisation
    # ────────────────────────────────────────────────────────────────────
    def _load_fault_and_grid(self) -> None:
        """Initialise RegularGrid3D, process fault file, integrate results."""
        # ── 1. Initialise grid ───────────────────────────────────────────
        logger.info("Initialising RegularGrid3D …")
        self.grid = RegularGrid3D(**GRID_CFG)
        logger.info(self.grid.summary())

        # ── 2. Process fault file ────────────────────────────────────────
        if not os.path.isfile(FAULT_FILE):
            logger.error(f"Fault file not found: {FAULT_FILE}")
            return

        logger.info("Processing fault file: %s", FAULT_FILE)
        try:
            fp, ds, ads = preprocess_fault(
                FAULT_FILE,
                offset_distance=200.0,
                grid_resolution=50.0,
                z_step=100.0,
                resample_spacing=50.0,
                smooth_window=5,
                max_turn_angle=60.0,
            )
        except Exception as exc:
            logger.exception(f"Fault processing failed: {exc}")
            return

        logger.info(
            "Fault arrays → fault_plane: %d pts | dip_side: %d pts | "
            "anti_dip_side: %d pts",
            len(fp), len(ds), len(ads),
        )

        # ── 3. Integrate into grid ────────────────────────────────────────
        logger.info("Integrating fault arrays into grid …")
        self.grid.integrate(fp,  label="fault_plane",    value=0.5)
        self.grid.integrate(ds,  label="dip_side",       value=1.0)
        self.grid.integrate(ads, label="anti_dip_side",  value=0.8)
        logger.info("Grid summary after integration:\n%s", self.grid.summary())

    # ────────────────────────────────────────────────────────────────────
    # UI Construction
    # ────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # --- Main horizontal pane ---
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # LEFT – Matplotlib visualisation
        left_frame = ttk.Frame(pane)
        pane.add(left_frame, weight=3)
        self._build_plot(left_frame)

        # RIGHT – Controls
        right_frame = ttk.Frame(pane, width=360)
        pane.add(right_frame, weight=1)
        self._build_controls(right_frame)

    # ── 3-D Plot ────────────────────────────────────────────────────────
    def _build_plot(self, parent: ttk.Frame) -> None:
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Depth (m)")
        self.ax.set_title("Well Planning View")

        # Invert Z so depth increases downward
        self.ax.invert_zaxis()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Connect pick event for clicking on wells
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    # ── Right-side controls ─────────────────────────────────────────────
    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        row = 0

        # ── Section: Wellhead Location ──────────────────────────────────
        sec_wh = ttk.LabelFrame(parent, text="Wellhead Platform Location",
                                padding=10)
        sec_wh.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 4))
        row += 1

        ttk.Label(sec_wh, text="X (m):").grid(row=0, column=0, sticky="w")
        self.entry_x = ttk.Entry(sec_wh, width=12)
        self.entry_x.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(sec_wh, text="Y (m):").grid(row=1, column=0, sticky="w")
        self.entry_y = ttk.Entry(sec_wh, width=12)
        self.entry_y.grid(row=1, column=1, padx=4, pady=2)

        btn_wh = ttk.Button(sec_wh, text="Set Wellhead",
                            command=self._on_set_wellhead)
        btn_wh.grid(row=2, column=0, columnspan=2, pady=(6, 0))

        # ── Section: Instructions ───────────────────────────────────────
        sec_info = ttk.LabelFrame(parent, text="Instructions", padding=10)
        sec_info.grid(row=row, column=0, sticky="ew", padx=8, pady=4)
        row += 1

        info_text = (
            "Instructions: < TBC >\n"
        )
        ttk.Label(sec_info, text=info_text, justify=tk.LEFT,
                  wraplength=320).grid(row=0, column=0, sticky="w")

        # ── Section: Selected wells & distributions ─────────────────────
        sec_sel = ttk.LabelFrame(parent,
                                 text="Selected Wells & Distributions",
                                 padding=10)
        sec_sel.grid(row=row, column=0, sticky="nsew", padx=8, pady=4)
        parent.rowconfigure(row, weight=1)
        row += 1

        # Treeview for distribution data
        cols = ("unit", "depth_from", "depth_to", "mean", "var")
        self.tree = ttk.Treeview(sec_sel, columns=cols, show="tree headings",
                                 height=12)
        self.tree.heading("#0", text="Well", anchor="w")
        self.tree.column("#0", width=90, minwidth=70)
        self.tree.heading("unit", text="Unit")
        self.tree.column("unit", width=50, anchor="center")
        self.tree.heading("depth_from", text="From (m)")
        self.tree.column("depth_from", width=65, anchor="e")
        self.tree.heading("depth_to", text="To (m)")
        self.tree.column("depth_to", width=60, anchor="e")
        self.tree.heading("mean", text="μ (m)")
        self.tree.column("mean", width=55, anchor="e")
        self.tree.heading("var", text="σ² (m²)")
        self.tree.column("var", width=60, anchor="e")

        scrollbar = ttk.Scrollbar(sec_sel, orient="vertical",
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Section: Run button ─────────────────────────────────────────
        btn_run = ttk.Button(parent, text="▶  Run Optimisation",
                             command=self._on_run)
        btn_run.grid(row=row, column=0, pady=12, ipady=6)

        # ── Section: Status bar ─────────────────────────────────────────
        row += 1
        _init_status = "Ready."
        self.status_var = tk.StringVar(value=_init_status)
        status_bar = ttk.Label(parent, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor="w", padding=4)
        status_bar.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 8))

    # ────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ────────────────────────────────────────────────────────────────────
    def _draw_wells(self) -> None:
        """Draw (or re-draw) all objects on the 3-D axes."""
        self.ax.cla()
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Depth (m)")
        self.ax.set_title("Well Planning View – G2-TKN Fault")
        self.ax.invert_zaxis()

        # Set axis limits to the fault / grid bounds
        self.ax.set_xlim(GRID_CFG["x_min"], GRID_CFG["x_max"])
        self.ax.set_ylim(GRID_CFG["y_min"], GRID_CFG["y_max"])
        self.ax.set_zlim(GRID_CFG["z_max"], 0)   # depth positive-down

        # ── Fault bands – plotted from grid cell centres, not raw arrays ──
        if self.grid is not None and self.grid.n_active > 0:
            act  = self.grid.active          # (nx, ny, nz) bool
            vals = self.grid.values          # (nx, ny, nz) float32
            xc   = self.grid.ds.coords["x"].values
            yc   = self.grid.ds.coords["y"].values
            zc   = self.grid.ds.coords["z"].values

            ix, iy, iz = np.where(act)
            gx = xc[ix];  gy = yc[iy];  gz = zc[iz]
            gv = vals[ix, iy, iz]

            def _scatter_grid_band(mask: np.ndarray, color: str,
                                   label: str, alpha: float,
                                   size: float) -> None:
                bx, by, bz = gx[mask], gy[mask], gz[mask]
                n = len(bx)
                if n == 0:
                    return
                if n > _MAX_DISP:
                    rng_idx = np.random.choice(n, _MAX_DISP, replace=False)
                    bx, by, bz = bx[rng_idx], by[rng_idx], bz[rng_idx]
                self.ax.scatter(bx, by, bz, c=color, s=size, alpha=alpha,
                                linewidths=0, label=label, zorder=2)

            # Value thresholds match integrate() calls:
            #   fault_plane → 0.5  |  anti_dip_side → 0.8  |  dip_side → 1.0
            # Overlapping cells carry the max value, so thresholds are unambiguous.
            _scatter_grid_band(gv < 0.6,                 CLR_FAULT_PLANE,   "Fault plane",   0.20, 3)
            _scatter_grid_band((gv >= 0.6) & (gv < 0.9), CLR_ANTI_DIP_SIDE, "Anti-dip side", 0.35, 5)
            _scatter_grid_band(gv >= 0.9,                CLR_DIP_SIDE,      "Dip side",      0.35, 5)

        # ── Well markers at surface (z=0) ───────────────────────────────
        for well in WELL_LOCATIONS:
            selected = well["name"] in self.selected_wells
            clr = CLR_WELL_SELECTED if selected else CLR_WELL_DEFAULT
            self.ax.scatter(
                well["x"], well["y"], 0,
                c=clr, s=MARKER_SIZE, marker="^",
                edgecolors="k", linewidths=0.5,
                picker=True, pickradius=8,
                label=well["name"],
                zorder=5,
            )
            self.ax.text(well["x"], well["y"], -80, well["name"],
                         fontsize=7, ha="center", va="top", zorder=5)

        # ── Wellhead marker ─────────────────────────────────────────────
        if self.wellhead_xy is not None:
            wx, wy = self.wellhead_xy
            self.ax.scatter(
                wx, wy, 0,
                c=CLR_WELLHEAD, s=150, marker="*",
                edgecolors="k", linewidths=0.6,
                zorder=6,
            )
            self.ax.text(wx, wy, -100, "Wellhead",
                         fontsize=8, fontweight="bold",
                         ha="center", va="top", color=CLR_WELLHEAD, zorder=6)

        # ── Optimised well path ─────────────────────────────────────────
        if self.well_path is not None:
            self.ax.plot(
                self.well_path[:, 0],
                self.well_path[:, 1],
                self.well_path[:, 2],
                color=CLR_WELL_PATH, linewidth=2.0,
                label="Well Path", zorder=4,
            )

        # ── Legend (deduplicated) ───────────────────────────────────────
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fault_keys = {"Fault plane", "Dip side", "Anti-dip side", "Well Path"}
        legend_items = {k: v for k, v in by_label.items() if k in fault_keys}
        if legend_items:
            self.ax.legend(legend_items.values(), legend_items.keys(),
                           loc="upper right", fontsize=7, markerscale=2)

        self.canvas.draw_idle()

    def _refresh_tree(self) -> None:
        """Rebuild the distribution treeview from current selection."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        for name, well in sorted(self.selected_wells.items()):
            parent_id = self.tree.insert(
                "", tk.END, text=f"{name}  ({well['x']}, {well['y']})",
                open=True,
            )
            for u in well["units"]:
                self.tree.insert(
                    parent_id, tk.END, text="",
                    values=(u["name"],
                            f"{u['depth_from']:.0f}",
                            f"{u['depth_to']:.0f}",
                            f"{u['thickness_mean']:.1f}",
                            f"{u['thickness_var']:.1f}"),
                )

    # ────────────────────────────────────────────────────────────────────
    # Event handlers
    # ────────────────────────────────────────────────────────────────────
    def _on_pick(self, event) -> None:
        """Handle click on a well marker – toggle selection."""
        if event.artist is None:
            return

        # Identify which well was clicked via the label
        label = event.artist.get_label()
        well = next((w for w in WELL_LOCATIONS if w["name"] == label), None)
        if well is None:
            return

        # Toggle
        if well["name"] in self.selected_wells:
            del self.selected_wells[well["name"]]
            self.status_var.set(f"Deselected {well['name']}")
        else:
            self.selected_wells[well["name"]] = well
            self.status_var.set(f"Selected {well['name']}")

        self._draw_wells()
        self._refresh_tree()

    def _on_set_wellhead(self) -> None:
        """Set wellhead platform location from text entries."""
        try:
            x = float(self.entry_x.get())
            y = float(self.entry_y.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Please enter numeric X and Y values.")
            return

        self.wellhead_xy = (x, y)
        self.well_path = None          # clear previous path
        self.status_var.set(f"Wellhead set at ({x:.1f}, {y:.1f})")
        self._draw_wells()

    def _on_run(self) -> None:
        """Run mock optimisation – generates a vertical well path."""
        if self.wellhead_xy is None:
            messagebox.showwarning("No wellhead",
                                   "Set the wellhead location first.")
            return
        if not self.selected_wells:
            messagebox.showwarning("No wells selected",
                                   "Select at least one offset well by "
                                   "clicking on its marker in the 3-D view.")
            return

        # ── Mock optimisation: straight vertical well path ──────────────
        wx, wy = self.wellhead_xy
        self.well_path = generate_vertical_well_path(wx, wy,
                                                     max_depth=4000.0,
                                                     step=10.0)
        self.status_var.set(
            f"Optimisation complete – vertical path at "
            f"({wx:.1f}, {wy:.1f}), 0-4 000 m"
        )
        self._draw_wells()


# ────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PathSightApp()
    app.mainloop()
