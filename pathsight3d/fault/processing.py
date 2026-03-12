"""Fault surface reconstruction and reservoir growth from Petrel fault files.

Public API
----------
pre_process_fault(filepath, **kwargs) -> (fault_plane, dip_side, anti_dip_side)

    fault_plane   : (N, 3) float32 array – dense interpolated fault surface (x, y, z)
    dip_side      : (M, 3) float32 array – reservoir band on the dip side of the fault
    anti_dip_side : (K, 3) float32 array – reservoir band on the anti-dip side

All Z values are positive-down depth (metres).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import uniform_filter1d
from shapely.geometry import LineString
from logging import getLogger

logger = getLogger(__name__)


# ── private helpers ────────────────────────────────────────────────────────────

def _load_petrel_fault(filepath: str) -> np.ndarray:
    """Parse a Petrel fault text file → (N, 3) float32 array [X, Y, Z].

    Z is always returned as positive depth even if the file stores it negative.
    """
    data: list[list[float]] = []
    in_data = False
    with open(filepath) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "END HEADER":
                in_data = True
                continue
            if not in_data or not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    pts = np.array(data, dtype=np.float32)
    if len(pts):
        pts[:, 2] = np.abs(pts[:, 2])   # positive-down convention
    return pts


def _resample_curve(xy: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a 2-D curve to uniform arc-length spacing."""
    if spacing <= 0 or len(xy) < 4:
        return xy
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < spacing * 3:
        return xy
    n_pts = max(10, int(total / spacing))
    s_new = np.linspace(0.0, total, n_pts)
    return np.column_stack([np.interp(s_new, s, xy[:, c]) for c in range(2)])


def _smooth_curve(xy: np.ndarray, window: int) -> np.ndarray:
    """Apply a uniform moving-average filter along each axis."""
    if window <= 1:
        return xy
    return np.column_stack(
        [uniform_filter1d(xy[:, c], window, mode="nearest") for c in range(2)]
    )


def _trim_tips(
    xy: np.ndarray,
    max_angle_deg: float = 60.0,
    scan_fraction: float = 0.15,
) -> np.ndarray:
    """Trim sharp artefacts from both tips of a contour curve.

    Walks inward from each tip removing segments whose local direction change
    exceeds *max_angle_deg*.  Never trims more than 25 % from either end.
    """
    if len(xy) < 20:
        return xy

    d = np.diff(xy, axis=0)
    angles = np.arctan2(d[:, 1], d[:, 0])
    raw_turn = np.abs(np.diff(angles))
    turn_deg = np.degrees(np.minimum(raw_turn, 2 * np.pi - raw_turn))

    n = len(xy)
    scan_n = max(5, int(n * scan_fraction))

    start = 0
    for k in range(min(scan_n, len(turn_deg))):
        if turn_deg[k] > max_angle_deg:
            start = k + 2
        else:
            break

    end = n
    for k in range(len(turn_deg) - 1, max(len(turn_deg) - scan_n - 1, -1), -1):
        if k < len(turn_deg) and turn_deg[k] > max_angle_deg:
            end = k
        else:
            break

    start = min(start, n // 4)
    end = max(end, 3 * n // 4)
    return xy if end <= start + 10 else xy[start:end]


def _label_dip_side(xy: np.ndarray, interp: LinearNDInterpolator) -> str:
    """Return ``'left'`` or ``'right'`` indicating which buffered side is dip.

    The dip direction is the direction in which depth **increases** fastest
    (steepest descent).  We compute the Z-gradient at the mid-point of the
    contour and project it onto the two normal directions.
    """
    mid = len(xy) // 2
    xc, yc = xy[mid]
    tan_x = xy[min(mid + 1, len(xy) - 1), 0] - xy[max(mid - 1, 0), 0]
    tan_y = xy[min(mid + 1, len(xy) - 1), 1] - xy[max(mid - 1, 0), 1]
    norm = max(np.hypot(tan_x, tan_y), 1e-12)
    tan_x /= norm
    tan_y /= norm

    dstep = 10.0
    zc  = float(interp(np.array([[xc,         yc        ]]))[0])
    zx  = float(interp(np.array([[xc + dstep, yc        ]]))[0])
    zy  = float(interp(np.array([[xc,         yc + dstep]]))[0])

    # gradient direction of Z (positive = depth increasing = dip direction)
    gx = (zx - zc) / dstep if np.isfinite(zx) and np.isfinite(zc) else 0.0
    gy = (zy - zc) / dstep if np.isfinite(zy) and np.isfinite(zc) else 0.0

    # normals to the tangent: left = (-tan_y, tan_x), right = (tan_y, -tan_x)
    dot_left  = gx * (-tan_y) + gy * tan_x
    dot_right = gx *   tan_y  + gy * (-tan_x)
    return "left" if dot_left > dot_right else "right"


def _extract_contours(
    Xg: np.ndarray,
    Yg: np.ndarray,
    Zg: np.ndarray,
    z_levels: np.ndarray,
) -> dict[float, np.ndarray]:
    """Batch-extract 2-D (x, y) contours at each depth level using matplotlib."""
    fig, ax = plt.subplots()
    cs = ax.contour(Xg, Yg, Zg, levels=z_levels)
    plt.close(fig)

    contours: dict[float, np.ndarray] = {}
    for i, z_lev in enumerate(z_levels):
        if i >= len(cs.allsegs):
            continue
        segs = [s for s in cs.allsegs[i] if len(s) >= 10]
        if segs:
            contours[float(z_lev)] = np.asarray(max(segs, key=len))
    return contours


def _polygon_to_xyz(polygon, z: float) -> np.ndarray | None:
    """Convert a shapely Polygon exterior to an (N, 3) xyz array at depth *z*."""
    if polygon is None or polygon.is_empty:
        return None
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)
    if polygon.geom_type != "Polygon":
        return None
    coords = np.array(polygon.exterior.coords)
    return np.column_stack([coords[:, 0], coords[:, 1], np.full(len(coords), z)])


# ── public API ─────────────────────────────────────────────────────────────────

def preprocess_fault(
    filepath: str,
    *,
    offset_distance: float = 200.0,
    grid_resolution: float = 50.0,
    z_step: float = 50.0,
    resample_spacing: float = 50.0,
    smooth_window: int = 5,
    max_turn_angle: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Petrel fault file and produce dense (x, y, z) point arrays.

    The sparse picks are interpolated onto a regular (x, y) grid to form a
    continuous fault surface.  The reservoir is then grown laterally by
    *offset_distance* metres away from the fault trace at each depth level,
    yielding a dip-side band and an anti-dip-side band.

    Parameters
    ----------
    filepath
        Path to a Petrel fault text file (``X Y Z`` columns after
        ``END HEADER``).
    offset_distance
        Lateral distance (metres) to buffer away from the fault trace.
    grid_resolution
        Spacing (metres) of the interpolation mesh in x and y.
    z_step
        Depth increment (metres) between successive contour levels.
    resample_spacing
        Arc-length resampling interval (metres) applied to each contour.
    smooth_window
        Moving-average half-window used to smooth each contour.
    max_turn_angle
        Maximum allowed direction change (degrees) before tip-trimming
        removes tail artefacts.

    Returns
    -------
    fault_plane : (N, 3) ndarray
        Dense, interpolated fault-surface points  (x, y, z).
    dip_side : (M, 3) ndarray
        Reservoir-growth band on the **dip** (downthrown) side.
    anti_dip_side : (K, 3) ndarray
        Reservoir-growth band on the **anti-dip** (upthrown) side.
    """
    # ── 1. load raw picks ─────────────────────────────────────────────────────
    pts = _load_petrel_fault(filepath)
    if len(pts) < 4:
        raise ValueError(
            f"Too few valid points in {filepath!r}: found {len(pts)}, need >= 4."
        )

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    logger.info("Loaded %d fault points from %r", len(pts), filepath)

    # ── 2. interpolate sparse picks → dense fault surface ────────────────────
    interp = LinearNDInterpolator(pts[:, :2], z)

    xg = np.arange(x.min(), x.max() + grid_resolution, grid_resolution)
    yg = np.arange(y.min(), y.max() + grid_resolution, grid_resolution)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = interp(Xg, Yg)

    valid = np.isfinite(Zg)
    if not valid.any():
        raise ValueError("Fault surface interpolation produced no valid values.")

    z_min = float(Zg[valid].min())
    z_max = float(Zg[valid].max())
    logger.info("  Interpolated depth range: [%.0f, %.0f] m", z_min, z_max)

    # Fault plane: all valid (x, y, z) nodes on the interpolated grid
    fault_plane = np.column_stack([Xg[valid], Yg[valid], Zg[valid]])

    # ── 3. extract horizontal contours at each depth level ────────────────────
    z_levels = np.arange(
        np.ceil(z_min / z_step) * z_step,
        np.floor(z_max / z_step) * z_step + z_step,
        z_step,
    )
    if len(z_levels) == 0:
        logger.warning("  No contour levels within depth range – returning fault plane only.")
        empty = np.empty((0, 3), dtype=np.float32)
        return fault_plane, empty, empty

    logger.info("  Extracting %d contour levels ...", len(z_levels))
    raw_contours = _extract_contours(Xg, Yg, Zg, z_levels)
    logger.info("  Valid contours extracted: %d / %d", len(raw_contours), len(z_levels))

    # ── 4. per-contour: clean, buffer, label sides → xyz point clouds ─────────
    dip_pts_list:  list[np.ndarray] = []
    anti_pts_list: list[np.ndarray] = []

    for z_lev, xy in sorted(raw_contours.items()):
        # clean the 2-D contour
        xy = _resample_curve(xy, resample_spacing)
        xy = _smooth_curve(xy, smooth_window)
        xy = _trim_tips(xy, max_turn_angle)

        if len(xy) < 6:
            continue

        # single-sided buffers on each side of the fault trace
        line = LineString(xy)
        buf_left  = line.buffer( offset_distance, single_sided=True,
                                  cap_style="flat", join_style="round")
        buf_right = line.buffer(-offset_distance, single_sided=True,
                                  cap_style="flat", join_style="round")

        # identify which side is dip (depth increases)
        dip_label = _label_dip_side(xy, interp)
        buf_dip   = buf_left  if dip_label == "left"  else buf_right
        buf_anti  = buf_right if dip_label == "left"  else buf_left

        dip_xyz  = _polygon_to_xyz(buf_dip,  z_lev)
        anti_xyz = _polygon_to_xyz(buf_anti, z_lev)

        if dip_xyz  is not None:
            dip_pts_list.append(dip_xyz)
        if anti_xyz is not None:
            anti_pts_list.append(anti_xyz)

    dip_side      = np.vstack(dip_pts_list)  if dip_pts_list  else np.empty((0, 3), dtype=np.float32)
    anti_dip_side = np.vstack(anti_pts_list) if anti_pts_list else np.empty((0, 3), dtype=np.float32)

    logger.info(
        "  fault_plane: %d pts | dip_side: %d pts | anti_dip_side: %d pts",
        len(fault_plane), len(dip_side), len(anti_dip_side),
    )
    return fault_plane, dip_side, anti_dip_side
