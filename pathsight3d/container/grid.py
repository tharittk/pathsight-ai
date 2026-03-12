"""Core data containers for PathSight3D."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr


class RegularGrid3D:
    """
    Regularized 3D grid for geobody data, backed by an xarray.Dataset.

    The dataset uses actual world coordinates as dimension labels and
    stores two data variables:
        - values: float32 property values  (dims: x, y, z)
        - active: bool cell mask           (dims: x, y, z)

    Constructor signature is identical to the previous dataclass version
    for full backward compatibility.

    Discarded vs. previous version (replaced by xarray coordinate system):
        - xyz_to_ijk   → ds.sel(x=..., y=..., z=..., method='nearest')
        - ijk_to_xyz   → coordinates are embedded in ds.coords
        - ijk_in_bounds → coordinate bounds are implicit in the Dataset
        - flat_index / unravel → work with labeled coordinates instead
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        dx: float,
        dy: float,
        dz: float,
    ):
        x_coords = x_min + np.arange(int((x_max - x_min) / dx) + 1) * dx
        y_coords = y_min + np.arange(int((y_max - y_min) / dy) + 1) * dy
        z_coords = z_min + np.arange(int((z_max - z_min) / dz) + 1) * dz

        # initialize as zero and inactive
        values = np.zeros(
            (len(x_coords), len(y_coords), len(z_coords)), dtype=np.float32
        )
        active = np.zeros_like(values, dtype=bool)

        self.ds = xr.Dataset(
            {"values": (["x", "y", "z"], values), "active": (["x", "y", "z"], active)},
            coords={"x": x_coords, "y": y_coords, "z": z_coords},
        )
        self._dx = float(dx)
        self._dy = float(dy)
        self._dz = float(dz)

    # --- Scalar grid parameters ---
    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dy(self) -> float:
        return self._dy

    @property
    def dz(self) -> float:
        return self._dz

    @property
    def nx(self) -> int:
        return self.ds.sizes["x"]

    @property
    def ny(self) -> int:
        return self.ds.sizes["y"]

    @property
    def nz(self) -> int:
        return self.ds.sizes["z"]

    @property
    def x_origin(self) -> float:
        return float(self.ds.coords["x"].values[0])

    @property
    def y_origin(self) -> float:
        return float(self.ds.coords["y"].values[0])

    @property
    def z_origin(self) -> float:
        return float(self.ds.coords["z"].values[0])

    # --- Data accessors ---
    # Return numpy arrays so that existing code (e.g. values[active]) keeps working.

    @property
    def values(self) -> np.ndarray:
        return self.ds["values"].values

    @values.setter
    def values(self, v: np.ndarray):
        self.ds["values"] = xr.DataArray(
            v.astype(np.float32), dims=["x", "y", "z"], coords=self.ds.coords
        )

    @property
    def active(self) -> np.ndarray:
        return self.ds["active"].values

    @active.setter
    def active(self, v: np.ndarray):
        self.ds["active"] = xr.DataArray(
            v.astype(bool), dims=["x", "y", "z"], coords=self.ds.coords
        )

    @property
    def n_active(self) -> int:
        return int(self.ds["active"].sum())

    @property
    def n_total(self) -> int:
        return self.nx * self.ny * self.nz

    def summary(self) -> str:
        xc = self.ds.coords["x"].values
        yc = self.ds.coords["y"].values
        zc = self.ds.coords["z"].values

        return (
            f"RegularGrid3D: {self.nx} x {self.ny} x {self.nz} = {self.n_total:,} cells\n"
            f"  X: [{xc[0]:.1f}, {xc[-1]:.1f}], dx={self._dx:.2f}\n"
            f"  Y: [{yc[0]:.1f}, {yc[-1]:.1f}], dy={self._dy:.2f}\n"
            f"  Z: [{zc[0]:.1f}, {zc[-1]:.1f}], dz={self._dz:.2f}\n"
            f"  Data variables: {', '.join(self.ds.data_vars.keys())}\n"
        )

    def integrate(
        self,
        xyz: np.ndarray,
        label: str,
        value: float = 1.0,
    ) -> None:
        """Map an (N, 3) array of world-coordinate points onto grid cells in-place.

        Each point in *xyz* is snapped to the nearest cell along every axis.
        Matched cells are activated and set to *value*; already-active cells
        receive ``max(existing_value, value)`` so that overlapping datasets
        blend by taking the higher value.

        Parameters
        ----------
        label : str
            Human-readable name used in log output.
        xyz : (N, 3) ndarray
            World-coordinate points  (x, y, z).  Z should be positive depth.
        value : float, optional
            Scalar written to every matched cell (default 1.0).
        """
        if len(xyz) == 0:
            return

        xc = self.ds.coords["x"].values
        yc = self.ds.coords["y"].values
        zc = self.ds.coords["z"].values

        # Snap each world coordinate to nearest grid index (vectorised)
        ix = np.clip(
            np.round((xyz[:, 0] - xc[0]) / self.dx).astype(int), 0, self.nx - 1
        )
        iy = np.clip(
            np.round((xyz[:, 1] - yc[0]) / self.dy).astype(int), 0, self.ny - 1
        )
        iz = np.clip(
            np.round((xyz[:, 2] - zc[0]) / self.dz).astype(int), 0, self.nz - 1
        )

        # Deduplicate cell indices
        stacked = np.column_stack([ix, iy, iz])
        unique_ijk = np.unique(stacked, axis=0)

        vals = self.values.copy()
        act = self.active.copy()

        for i, j, k in unique_ijk:
            if not act[i, j, k]:
                act[i, j, k] = True
                vals[i, j, k] = float(value)
            else:
                vals[i, j, k] = max(float(vals[i, j, k]), float(value))

        self.active = act
        self.values = vals

        print(
            f"  integrate({label!r}): {len(xyz):,} input pts "
            f"-> {len(unique_ijk):,} cells activated/updated  "
            f"(total active: {self.n_active:,})"
        )
