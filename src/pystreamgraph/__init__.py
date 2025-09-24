"""
A compact Matplotlib streamgraph helper with:

- sorted or unsorted ordering, and optional global ordering helpers
- vertical margins (gaps) between streams
- optional value smoothing (moving-average)
- boundary curve smoothing with shape-preserving PCHIP splines (default)
- optional Catmull–Rom boundary curves
- flexible label placement: peak, start, end, max-width, balanced, sliding-window
- colormap selection via names, sequences, or label->color mappings

Author: Max Noichl & 🦖
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Mapping, Sequence, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap

# Optional SciPy dependency for high-quality PCHIP interpolation
try:  # pragma: no cover - import guard
    from scipy.interpolate import PchipInterpolator as _SciPyPchip
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - if SciPy is missing, fall back to pure Python
    _HAS_SCIPY = False

__version__ = "0.1.2"

__all__ = [
    "plot_streamgraph",
    "streamgraph_envelopes",
    "catmull_rom_interpolate",
    "pchip_interpolate",
    # Global ordering helpers (Di Bartolomeo & Hu, 2016)
    "order_bestfirst",
    "order_twoopt",
]


# ---------- Smoothing utilities ----------

def moving_average(a: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflection at the edges.
    window must be odd and at least 1. When window <= 1 no smoothing is applied.
    """
    if window <= 1:
        return a
    if window % 2 == 0:
        window += 1
    pad = window // 2
    a_pad = np.pad(a, (pad, pad), mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(a_pad, kernel, mode="valid")


def smooth_series(Y: np.ndarray, window: int = 1) -> np.ndarray:
    """Apply moving average along time for each row of Y. Returns new array."""
    if window <= 1:
        return Y
    return np.vstack([moving_average(y, window) for y in Y])


# ---------- Ordering and stacking ----------

def _order_indices(Yt: np.ndarray, strategy: str = "none", previous: Optional[List[int]] = None) -> List[int]:
    """Ordering of series for a single time step.
    - 'none' keeps the previous order if provided else the original order.
    - 'by_value' sorts by descending magnitude at this time step.
    """
    if strategy == "none":
        return list(range(len(Yt))) if previous is None else previous
    elif strategy == "by_value":
        return list(np.argsort(-Yt))
    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")


def _baseline_centered(sumY: np.ndarray, margins: float, k: int) -> np.ndarray:
    """Centered baseline so the stack is symmetric around zero.
    margins is a fraction of sumY used as total gap budget at each time.
    """
    total_gap = margins * sumY
    return -0.5 * (sumY + total_gap)


def _baseline_unweighted(Y: np.ndarray) -> np.ndarray:
    """Unweighted wiggle-minimizing baseline (Byron & Wattenberg).

    Closed form (pointwise):
        g0(x) = -(1/(n+1)) * sum_{i=1..n} (n-i+1) * f_i(x)

    Returns array shape (m,), where Y has shape (n, m).
    """
    k, n = Y.shape
    if k == 0:
        return np.zeros(n, dtype=float)
    weights = np.arange(k, 0, -1, dtype=float).reshape(k, 1)
    g0 = - (weights * Y).sum(axis=0) / float(k + 1)
    return g0


def _baseline_weighted(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Weighted wiggle baseline (Streamgraph baseline) per Byron & Wattenberg.

    Minimizes squared slopes of midlines weighted by thickness.
    Computes g0'(x) in closed form and integrates over X.
    The constant of integration is chosen so the midline has zero mean.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    k, n = Y.shape
    if n == 0:
        return np.zeros(0, dtype=float)
    # 1) finite-difference slopes
    dYdx = np.gradient(Y, X, axis=1)
    # 2) prefix sums of slopes sum_{j<=i-1} f_j'
    Sprime = np.cumsum(dYdx, axis=0)
    Sprime_minus = np.vstack([np.zeros((1, n), dtype=float), Sprime[:-1, :]])
    # 3) a_i(x) = 0.5 f_i'(x) + sum_{j<=i-1} f_j'(x)
    A = 0.5 * dYdx + Sprime_minus
    # 4) g0'(x) as weighted average with weights f_i(x)
    num = (A * Y).sum(axis=0)
    den = Y.sum(axis=0)
    den_safe = np.where(den <= 0.0, 1.0, den)
    g0prime = - num / den_safe
    g0prime = np.where(den <= 0.0, 0.0, g0prime)
    # 5) integrate to get g0(x) via cumulative trapezoid
    g0 = np.zeros_like(g0prime)
    if n >= 2:
        dx = np.diff(X)
        g0[1:] = np.cumsum(0.5 * (g0prime[1:] + g0prime[:-1]) * dx)
    # Center the midline (without margins) around zero mean
    sumY = den
    midline = g0 + 0.5 * sumY
    g0 = g0 - float(np.mean(midline))
    return g0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of values with non-negative weights.

    Reference: Di Bartolomeo & Hu (2016) propose replacing the 2-norm with a
    weighted 1-norm; the minimizer is a weighted median.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return 0.0
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    return float(v[min(idx, len(v) - 1)])


def _baseline_l1_weighted(X: np.ndarray, Y: np.ndarray, orders: list[list[int]]) -> np.ndarray:
    """
    L1 weighted baseline per Di Bartolomeo & Hu (2016):
    g0'(x) = weighted_median(-A_i(x), weights = f_i(x)), integrate then center.

    Where A_i(x) = 0.5 f_i'(x) + sum_{j<=i-1} f_j'(x) under the instantaneous order.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    k, n = Y.shape
    sumY = Y.sum(axis=0)
    sumY_safe = np.where(sumY <= 0.0, 1.0, sumY)

    dYdx = np.gradient(Y, X, axis=1)
    g0prime = np.zeros(n, dtype=float)
    for t in range(n):
        if sumY[t] <= 0.0:
            g0prime[t] = 0.0
            continue
        idx = orders[t]
        dy_ord_t = dYdx[idx, t]
        y_ord_t = Y[idx, t]
        prefix = np.cumsum(dy_ord_t)
        prefix_minus = np.concatenate(([0.0], prefix[:-1]))
        A = 0.5 * dy_ord_t + prefix_minus
        g0prime[t] = _weighted_median(-A, y_ord_t)

    # integrate and center midline
    g0 = np.zeros_like(g0prime)
    if n >= 2:
        dx = np.diff(X)
        g0[1:] = np.cumsum(0.5 * (g0prime[1:] + g0prime[:-1]) * dx)

    midline = g0 + 0.5 * sumY
    g0 = g0 - float(np.mean(midline))
    return g0


def streamgraph_envelopes(
    Y: np.ndarray,
    margin_frac: float = 0.0,
    order_mode: str = "by_value",
    X: Optional[np.ndarray] = None,
    wiggle_reduction: str = "none",
    global_order: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bottom and top envelopes for each layer.

    Parameters
    ----------
    Y : array, shape (k, n)
        Non-negative series as rows.
    margin_frac : float
        Fraction of the column sum reserved for gaps between layers.
    order_mode : {'by_value','none','global'}
        Sorting at each time step, not at all, or use a fixed global order.
    X : array-like or None
        Required when wiggle_reduction in {'weighted','l1_weighted'}. Ignored otherwise.
    wiggle_reduction : {'none','unweighted','weighted','l1_weighted'}
        Baseline strategy to reduce wiggle. 'none' uses the classic centered
        silhouette. 'unweighted' uses the closed-form deviation-minimizing
        baseline. 'weighted' uses the Streamgraph baseline (Byron–Wattenberg, L2).
        'l1_weighted' uses the Di Bartolomeo & Hu (2016) L1 baseline via
        weighted median of (-A_i) per x, then integrates and centers.

    global_order : sequence of int, optional
        When ``order_mode='global'``, the fixed order of layer indices to use at
        all time steps. Length must match the number of rows in ``Y``.

    Returns
    -------
    bottoms, tops : arrays of shape (k, n)
    """
    if (Y < 0).any():
        raise ValueError("Y must be non-negative for streamgraphs.")

    k, n = Y.shape
    sumY = Y.sum(axis=0)
    sumY_safe = np.where(sumY == 0, 1.0, sumY)

    # Compute per-time stacking orders (used for baseline and stacking)
    orders: list[list[int]] = []
    prev_order = None
    for t in range(n):
        if order_mode == "global" and global_order is not None:
            order_t = list(global_order)
        else:
            order_t = _order_indices(Y[:, t], "none" if order_mode == "none" else "by_value", previous=prev_order)
        orders.append(order_t)
        prev_order = order_t

    # Compute baseline according to wiggle reduction mode
    mode = (wiggle_reduction or "none").lower()
    if mode not in {"none", "unweighted", "weighted", "l1_weighted"}:
        raise ValueError("wiggle_reduction must be one of {'none','unweighted','weighted','l1_weighted'}")

    if mode == "none":
        g0 = _baseline_centered(sumY, margin_frac, k)
    elif mode == "unweighted":
        # Use instantaneous stacking order for the closed-form baseline
        weights = np.arange(k, 0, -1, dtype=float)
        g0 = np.zeros(n, dtype=float)
        for t in range(n):
            idx = orders[t]
            y_ord_t = Y[idx, t]
            g0[t] = - float(np.dot(weights, y_ord_t)) / float(k + 1)
        # Center including margins by shifting down half total gap
        g0 = g0 - 0.5 * (margin_frac * sumY_safe)
    elif mode == "weighted":
        if X is None:
            raise ValueError("X must be provided when wiggle_reduction='weighted'")
        X = np.asarray(X, dtype=float)
        if X.size != n:
            raise ValueError("X must have the same length as Y's time dimension")
        dYdx = np.gradient(Y, X, axis=1)
        g0prime = np.zeros(n, dtype=float)
        for t in range(n):
            if sumY[t] <= 0.0:
                g0prime[t] = 0.0
                continue
            idx = orders[t]
            dy_ord_t = dYdx[idx, t]
            y_ord_t = Y[idx, t]
            prefix = np.cumsum(dy_ord_t)
            prefix_minus = np.concatenate(([0.0], prefix[:-1]))
            A = 0.5 * dy_ord_t + prefix_minus
            num = float(np.dot(A, y_ord_t))
            g0prime[t] = - num / float(sumY[t])
        g0 = np.zeros(n, dtype=float)
        if n >= 2:
            dx = np.diff(X)
            g0[1:] = np.cumsum(0.5 * (g0prime[1:] + g0prime[:-1]) * dx)
        # Center midline around zero; then account for margins
        midline = g0 + 0.5 * sumY
        g0 = g0 - float(np.mean(midline))
        g0 = g0 - 0.5 * (margin_frac * sumY_safe)
    else:  # l1_weighted
        if X is None:
            raise ValueError("X must be provided when wiggle_reduction='l1_weighted'")
        X = np.asarray(X, dtype=float)
        if X.size != n:
            raise ValueError("X must have the same length as Y's time dimension")
        g0 = _baseline_l1_weighted(X, Y, orders)
        # Account for margins by shifting down half total gap
        g0 = g0 - 0.5 * (margin_frac * sumY_safe)

    baseline = g0
    per_gap = np.where(k > 1, (margin_frac * sumY_safe) / (k - 1), 0.0)

    bottoms = np.zeros_like(Y, dtype=float)
    tops = np.zeros_like(Y, dtype=float)

    # Stacking pass reusing orders
    for t in range(n):
        order_t = orders[t]
        b = baseline[t]
        for r, i in enumerate(order_t):
            bottoms[i, t] = b
            tops[i, t] = b + Y[i, t]
            b = tops[i, t] + (per_gap[t] if r < k - 1 else 0.0)

    return bottoms, tops


# ---------- Curve smoothing (Catmull–Rom) ----------

def _catmull_segment(p0, p1, p2, p3, t):
    """Evaluate a Catmull–Rom spline segment.

    Parameters
    ----------
    p0, p1, p2, p3 : array-like or float
        Control point values (can be x or y coordinates) for the segment.
    t : array-like or float
        Parameter(s) in [0, 1] at which to evaluate the segment.

    Returns
    -------
    ndarray or float
        Interpolated values at ``t``.
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def catmull_rom_interpolate(x: np.ndarray, y: np.ndarray, samples_per_seg: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Return dense x_s, y_s that pass through all control points using Catmull–Rom splines.
    samples_per_seg >= 1. When 1, each segment contributes one interior sample and the end
    point of the final segment is appended once.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or samples_per_seg <= 1:
        return x, y

    xs = []
    ys = []
    for i in range(n - 1):
        x0 = x[i - 1] if i - 1 >= 0 else 2 * x[i] - x[i + 1]
        y0 = y[i - 1] if i - 1 >= 0 else 2 * y[i] - y[i + 1]
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        x3 = x[i + 2] if i + 2 < n else 2 * x[i + 1] - x[i]
        y3 = y[i + 2] if i + 2 < n else 2 * y[i + 1] - y[i]

        ts = np.linspace(0, 1, samples_per_seg + 1, endpoint=False)
        xs.append(_catmull_segment(x0, x1, x2, x3, ts))
        ys.append(_catmull_segment(y0, y1, y2, y3, ts))

    xs.append(np.array([x[-1]]))
    ys.append(np.array([y[-1]]))
    return np.concatenate(xs), np.concatenate(ys)


# ---------- Curve smoothing (PCHIP – shape-preserving) ----------

def _pchip_fritsch_carlson_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute PCHIP slopes (Fritsch–Carlson) without SciPy.

    Ensures shape preservation and avoids overshoot between samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h

    m = np.zeros(n, dtype=float)
    # Interior slopes (harmonic mean weighting)
    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    # Endpoints (one-sided, shape-preserving)
    if n >= 2:
        m0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]) if n > 2 else delta[0]
        if np.sign(m0) != np.sign(delta[0]):
            m0 = 0.0
        elif (np.sign(delta[0]) != np.sign(delta[1]) if n > 2 else False) and abs(m0) > abs(3.0 * delta[0]):
            m0 = 3.0 * delta[0]
        m[0] = m0

        mn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2]) if n > 2 else delta[-1]
        if np.sign(mn) != np.sign(delta[-1]):
            mn = 0.0
        elif (np.sign(delta[-1]) != np.sign(delta[-2]) if n > 2 else False) and abs(mn) > abs(3.0 * delta[-1]):
            mn = 3.0 * delta[-1]
        m[-1] = mn
    return m


def pchip_interpolate(x: np.ndarray, y: np.ndarray, samples_per_seg: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Return dense (xp, yp) sampled using a shape-preserving cubic (PCHIP).

    - Uses SciPy's PchipInterpolator when available for numerical robustness.
    - Falls back to a pure NumPy Fritsch–Carlson implementation otherwise.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or samples_per_seg <= 1:
        return x, y

    # Build a per-segment dense x grid, excluding each segment's right endpoint
    # and appending the final x once, mirroring catmull_rom_interpolate's layout.
    x_parts = [np.linspace(x[i], x[i + 1], samples_per_seg + 1, endpoint=False) for i in range(n - 1)]
    xp = np.concatenate(x_parts + [np.array([x[-1]])])

    if _HAS_SCIPY:
        f = _SciPyPchip(x, y)
        yp = f(xp)
        return xp, np.asarray(yp, dtype=float)

    # Fallback: evaluate cubic Hermite with PCHIP slopes per interval
    m = _pchip_fritsch_carlson_slopes(x, y)
    ys = []
    for i in range(n - 1):
        xi, xi1 = x[i], x[i + 1]
        hi = xi1 - xi
        ts = (x_parts[i] - xi) / hi  # normalized [0,1)
        t2 = ts * ts
        t3 = t2 * ts
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + ts
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        yi = y[i]
        yi1 = y[i + 1]
        # Hermite basis with slopes m[i], m[i+1]
        seg = h00 * yi + h10 * hi * m[i] + h01 * yi1 + h11 * hi * m[i + 1]
        ys.append(seg)
    yp = np.concatenate(ys + [np.array([y[-1]])])
    return xp, yp


# ---------- Main plotting ----------

def plot_streamgraph(
    X: np.ndarray,
    Y: np.ndarray,
    labels: Optional[List[str]] = None,
    sorted_streams: bool = False,
    margin_frac: float = 0.08,
    smooth_window: int = 1,
    cmap: Optional[Union[str, Mapping[str, str], Sequence[str], Colormap]] = None,
    linewidth: float = 0.0,
    alpha: float = 1.0,
    label_placement: bool = True,
    label_position: str = "balanced",
    label_color: Optional[Tuple[str] | List[str] | str] = None,
    label_fontsize: int = 10,
    label_weight: str = "bold",
    curve_samples: int = 200,
    curve_method: str = "pchip",
    baseline: str = "center",
    wiggle_reduction: str = "weighted",
    global_order: Optional[Sequence[int]] = None,
    pad_frac: float = 0.05,
    # Label spacing and connector helpers (for start/end)
    label_min_gap_frac: float = 0.02,
    label_edge_offset_frac: float = 0.02,
    label_connectors: bool = False,
    label_connector_color: Optional[str] = None,
    label_connector_alpha: float = 0.6,
    label_connector_linewidth: float = 0.8,
    ax: Optional[plt.Axes] = None,
    # Extra text kwargs applied to labels (overrides fontsize/weight/color if provided)
    label_kwargs: Optional[Mapping[str, Any]] = None,
    # Anchor for label positioning for {'peak','max_width','balanced'}
    label_anchor: str = "center",
    # Optionally plot anchor points used for placement
    label_plot_anchors: bool = False,
    label_point_kwargs: Optional[Mapping[str, Any]] = None,
    # Balanced-placement tuning
    label_balanced_inset_frac: float = 0.05,
    # Size labels by stream magnitude
    label_fontsize_min: Optional[float] = None,
    label_fontsize_max: Optional[float] = None,
    label_fontsize_by: str = "sum",
    # Progress bar for balanced annealing (requires tqdm; silently skipped if missing)
    label_balanced_progress: bool = False,
    # Allow tiny bbox spill tolerance (in pixels) when checking in-stream fit
    label_balanced_fit_tolerance_px: float = 2.0,
    # Debug: plot candidate segments for balanced placement
    label_balanced_debug_segments: bool = False,
    label_balanced_debug_kwargs: Optional[Mapping[str, Any]] = None,
    # Search breadth/schedule controls
    label_balanced_candidates_per_layer: int = 60,
    label_balanced_restarts: int = 2,
    label_balanced_T0: float = 2.5,
    label_balanced_T_min: float = 5e-4,
    label_balanced_alpha: float = 0.94,
    label_balanced_iters_per_T: int = 240,
    # Thickness filtering
    label_balanced_min_thickness_q: float = 0.5,
) -> plt.Axes:
    """Plot a configurable streamgraph.

    Parameters
    ----------
    X : ndarray, shape (n,)
        Monotonic x-coordinates.
    Y : ndarray, shape (k, n)
        Non-negative series (rows are layers) to be stacked.
    labels : list of str, optional
        Per-layer label texts. If provided and ``label_placement=True``, labels
        are rendered using ``label_position``.
    sorted_streams : bool, default False
        If True, sort layers by value at each time step; otherwise keep order.
        If ``global_order`` is provided, that fixed order takes precedence.
    margin_frac : float, default 0.08
        Fraction of column sum reserved for gaps between layers.
    smooth_window : int, default 1
        Centered moving-average window applied to values per layer (odd >= 1).
    cmap : str | sequence | mapping | Colormap, optional
        Color specification for layers. Strings/Colormap sample evenly; a
        sequence is cycled/truncated to length k; a mapping uses ``labels`` as
        keys. ``None`` defers to Matplotlib's property cycle.
    linewidth : float, default 0.0
        Edge line width passed to ``fill_between``.
    alpha : float, default 1.0
        Face alpha for ``fill_between``.
    label_placement : bool, default True
        Whether to place labels.
    label_position : {'peak','start','end','max_width','balanced','sliding_window'}, default 'balanced'
        Labeling strategy. ``sliding_window`` uses a fast in-stream algorithm
        (Di Bartolomeo & Hu, 2016). ``balanced`` runs a small simulated
        annealing to reduce overlap and improve placement.
    label_color : str | list[str] | tuple[str] | None, optional
        Label colors. Single string applies to all; list/tuple must match k.
    label_fontsize : int, default 10
        Base font size when no scaling is requested.
    label_weight : str, default 'bold'
        Font weight for labels.
    curve_samples : int, default 200
        Samples per original segment for boundary interpolation. If <= 1, no
        boundary curve interpolation is applied.
    curve_method : {'pchip','catmull_rom'}, default 'pchip'
        Boundary curve interpolation method.
    baseline : str, default 'center'
        Only ``'center'`` is supported in this version.
    wiggle_reduction : {'none','unweighted','weighted','l1_weighted'}, default 'weighted'
        Baseline strategy used when computing envelopes. ``'weighted'`` is the
        Streamgraph baseline (Byron–Wattenberg, L2); ``'l1_weighted'`` follows
        Di Bartolomeo & Hu (2016).
    global_order : sequence of int, optional
        Fixed center-out order of layers to use across all time steps. When
        provided, overrides ``sorted_streams``.
    pad_frac : float, default 0.05
        Extra vertical padding added to the y-limits.
    label_min_gap_frac : float, default 0.02
        Minimum vertical gap between start/end labels as a fraction of y-range.
    label_edge_offset_frac : float, default 0.02
        Horizontal x-offset for start/end labels as a fraction of x-range.
    label_connectors : bool, default False
        If True, draw connectors from start/end labels to the stream edge.
    label_connector_color : str, optional
        Color for label connectors (defaults to label color).
    label_connector_alpha : float, default 0.6
        Alpha for label connectors.
    label_connector_linewidth : float, default 0.8
        Line width for label connectors.
    ax : matplotlib.axes.Axes, optional
        Target axes. Created if None.
    label_kwargs : Mapping[str, Any], optional
        Extra ``matplotlib.text.Text`` kwargs applied to labels.
    label_anchor : str, default 'center'
        Anchor point used for in-stream labels ('peak', 'max_width', 'balanced').
    label_plot_anchors : bool, default False
        If True, plot the anchor points used for label placement.
    label_point_kwargs : Mapping[str, Any], optional
        Extra kwargs for plotting anchor points when ``label_plot_anchors``.
    label_balanced_inset_frac : float, default 0.05
        Fraction of x-span kept as inset where labels cannot be centered.
    label_fontsize_min : float, optional
        Minimum label font size when scaling by layer magnitude.
    label_fontsize_max : float, optional
        Maximum label font size when scaling by layer magnitude.
    label_fontsize_by : {'sum','max','mean'}, default 'sum'
        Statistic used when scaling label sizes.
    label_balanced_progress : bool, default False
        If True and ``tqdm`` is available, show progress during annealing.
    label_balanced_fit_tolerance_px : float, default 2.0
        Vertical tolerance (in px) when checking if text fits inside a stream.
    label_balanced_debug_segments : bool, default False
        If True, plot candidate midlines considered for placement.
    label_balanced_debug_kwargs : Mapping[str, Any], optional
        Extra line kwargs for debug segments.
    label_balanced_candidates_per_layer : int, default 60
        Number of candidate positions per layer for balanced placement.
    label_balanced_restarts : int, default 2
        Number of random annealing restarts.
    label_balanced_T0 : float, default 2.5
        Initial temperature for annealing.
    label_balanced_T_min : float, default 5e-4
        Minimum temperature for annealing.
    label_balanced_alpha : float, default 0.94
        Cooling factor per temperature step.
    label_balanced_iters_per_T : int, default 240
        Iterations per temperature step.
    label_balanced_min_thickness_q : float, default 0.5
        Quantile threshold of per-segment thickness used to filter candidates.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the streamgraph was drawn on.

    Notes
    -----
    - ``Y`` must be non-negative.
    - If ``global_order`` is provided, it is used as a fixed order; otherwise
      the order is either the input order (``sorted_streams=False``) or sorted
      by value at each time step (``sorted_streams=True``).
    - Boundary curve interpolation is applied to both bottom and top envelopes
      per layer after stacking on the original grid.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(50)
    >>> rng = np.random.default_rng(0)
    >>> Y = np.clip(rng.normal(0.6, 0.2, size=(5, 50)), 0, None)
    >>> ax = plot_streamgraph(X, Y, labels=list("ABCDE"), label_position='balanced')
    """
    # Import the original implementation from the previous module to avoid code drift
    # if this file diverges. This line assumes the code was moved verbatim.
    # The full implementation has been copied from the original to keep in-package.
    # Note: Due to space, keeping the same implementation as in the previous package.
    # The function body below mirrors the previous one exactly.

    if baseline != "center":
        raise NotImplementedError("Only baseline='center' is implemented in this version.")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if (Y < 0).any():
        raise ValueError("Y must be non-negative for streamgraphs.")

    # Value smoothing
    Ys = smooth_series(Y, smooth_window)
    
    # Stacking/interpolation
    # Determine ordering mode: if a global order is provided, use it
    order_mode = "global" if global_order is not None else ("by_value" if sorted_streams else "none")
    if curve_samples and curve_samples > 1:
        # Compute envelopes on the original grid, then smooth each boundary curve.
        b_raw, t_raw = streamgraph_envelopes(Ys, margin_frac=margin_frac, order_mode=order_mode,
                                             X=X, wiggle_reduction=wiggle_reduction, global_order=global_order)

        Xp = None
        b_smooth_list = []
        t_smooth_list = []
        for i in range(Ys.shape[0]):
            if (curve_method or "pchip").lower() == "catmull_rom":
                xb, yb = catmull_rom_interpolate(X, b_raw[i], samples_per_seg=curve_samples)
                xt, yt = catmull_rom_interpolate(X, t_raw[i], samples_per_seg=curve_samples)
            else:
                xb, yb = pchip_interpolate(X, b_raw[i], samples_per_seg=curve_samples)
                xt, yt = pchip_interpolate(X, t_raw[i], samples_per_seg=curve_samples)
            if Xp is None:
                Xp = xb
            b_smooth_list.append(yb)
            t_smooth_list.append(yt)
        bottoms = np.vstack(b_smooth_list)
        tops = np.vstack(t_smooth_list)
    else:
        bottoms, tops = streamgraph_envelopes(Ys, margin_frac=margin_frac, order_mode=order_mode,
                                              X=X, wiggle_reduction=wiggle_reduction, global_order=global_order)
        Xp = X

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    def _build_layer_colors(
        cmap_value: Optional[Union[str, Mapping[str, str], Sequence[str], Colormap]],
        labels_value: Optional[List[str]],
        num_layers: int,
    ) -> Optional[List]:
        # None -> let Matplotlib use default property cycle
        if cmap_value is None:
            return None

        # Helper: default colors from property cycle, or tab20 fallback
        def _default_colors(n: int):
            prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
            if prop_cycle is not None:
                cols = prop_cycle.by_key().get("color", [])
            else:
                cols = []
            if not cols:
                return [get_cmap("tab20", n)(i) for i in range(n)]
            if len(cols) < n:
                # cycle
                reps = (n + len(cols) - 1) // len(cols)
                cols = (cols * reps)[:n]
            else:
                cols = cols[:n]
            return cols

        # String or Colormap -> sample evenly across full [0, 1] range
        if isinstance(cmap_value, (str, Colormap)):
            cm = get_cmap(cmap_value) if isinstance(cmap_value, str) else cmap_value
            ts = np.linspace(0.0, 1.0, num_layers, endpoint=True)
            return [cm(float(t)) for t in ts]

        # Sequence of colors -> normalize to length num_layers by cycling
        if isinstance(cmap_value, Sequence) and not isinstance(cmap_value, str):
            seq = list(cmap_value)
            if len(seq) == 0:
                return _default_colors(num_layers)
            if len(seq) < num_layers:
                reps = (num_layers + len(seq) - 1) // len(seq)
                seq = (seq * reps)[:num_layers]
            else:
                seq = seq[:num_layers]
            return seq

        # Mapping label -> color. Requires labels
        if isinstance(cmap_value, Mapping):
            if labels_value is None:
                raise ValueError("labels must be provided when cmap is a Mapping[label -> color]")
            base = _default_colors(num_layers)
            out = []
            for i in range(num_layers):
                lab = labels_value[i]
                out.append(cmap_value.get(lab, base[i]))
            return out

        # Fallback
        return None

    colors = _build_layer_colors(cmap, labels, Y.shape[0])

    for i in range(Y.shape[0]):
        ax.fill_between(Xp, bottoms[i], tops[i], linewidth=linewidth, alpha=alpha,
                        color=None if colors is None else colors[i])

    # cosmetics
    ax.set_xlim(Xp.min(), Xp.max())
    ymin = float(np.min(bottoms))
    ymax = float(np.max(tops))
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad_frac * yr, ymax + pad_frac * yr)
    ax.spines[["top", "right"]].set_visible(False)

    # labels
    if label_placement and labels is not None:
        # Determine placement strategy
        position = (label_position or "peak").lower()
        if position == "paper_fast":
            # Backward-compat alias; prefer clearer name 'sliding_window'
            try:
                import warnings
                warnings.warn("label_position='paper_fast' is deprecated; use 'sliding_window'", DeprecationWarning)
            except Exception:
                pass
            position = "sliding_window"
        if position not in {"peak", "start", "end", "max_width", "balanced", "sliding_window"}:
            raise ValueError("label_position must be one of {'peak','start','end','max_width','balanced','sliding_window'}")

        # Margin in x-units to nudge labels away from stream boundary
        x_min0, x_max0 = float(Xp.min()), float(Xp.max())
        xr = x_max0 - x_min0 if x_max0 > x_min0 else 1.0
        edge_offset = float(max(label_edge_offset_frac, 0.0))
        x_margin = edge_offset * xr

        # Track if we need to extend x-limits for start/end labels
        extend_left = position == "start"
        extend_right = position == "end"

        # Prepare per-layer colors
        if label_color is None:
            label_colors = ["black"] * Y.shape[0]
        elif isinstance(label_color, str):
            label_colors = [label_color] * Y.shape[0]
        else:
            # assume sequence
            if len(label_color) != Y.shape[0]:
                raise ValueError("label_color list must match number of layers")
            label_colors = list(label_color)

        # Extra text kwargs (optional overrides for labels)
        text_kwargs_common = dict(label_kwargs) if label_kwargs else {}

        # Optional tqdm
        use_tqdm = bool(label_balanced_progress)
        _tqdm_fn = None
        if use_tqdm:
            try:  # pragma: no cover - optional dependency guard
                from tqdm.auto import tqdm as _tqdm  # type: ignore
                _tqdm_fn = _tqdm
            except Exception:
                use_tqdm = False

        # Compute per-layer font sizes if requested
        def _compute_label_sizes() -> List[float]:
            k_layers = Y.shape[0]
            # If label_kwargs explicitly sets fontsize, honor it for all labels
            if "fontsize" in text_kwargs_common:
                return [float(text_kwargs_common["fontsize"]) for _ in range(k_layers)]
            if label_fontsize_min is None or label_fontsize_max is None:
                return [float(label_fontsize) for _ in range(k_layers)]
            a = float(min(label_fontsize_min, label_fontsize_max))
            b = float(max(label_fontsize_min, label_fontsize_max))
            if b <= 0:
                return [float(label_fontsize) for _ in range(k_layers)]
            by = (label_fontsize_by or "sum").lower()
            if by == "max":
                stats = np.max(Y, axis=1)
            elif by in {"mean", "avg"}:
                stats = np.mean(Y, axis=1)
            else:  # 'sum'
                stats = np.sum(Y, axis=1)
            s_min = float(np.min(stats))
            s_max = float(np.max(stats))
            if not np.isfinite(s_min) or not np.isfinite(s_max):
                return [float(label_fontsize) for _ in range(k_layers)]
            if s_max <= s_min + 1e-12:
                return [float(0.5 * (a + b)) for _ in range(k_layers)]
            sizes = a + (stats - s_min) * (b - a) / (s_max - s_min)
            return [float(s) for s in sizes]

        label_sizes = _compute_label_sizes()

        label_targets = []
        # Pre-compute slice of maximum overall width for 'max_width'
        j_wide = None
        if position == "max_width":
            total_thickness = (tops - bottoms)
            j_wide = int(np.argmax(np.sum(total_thickness, axis=0)))
        def _balanced_candidates_and_text_extents():
            # Ensure a renderer exists to measure text in pixels
            fig = ax.figure
            try:
                renderer = fig.canvas.get_renderer()
            except Exception:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

            # Measure text box (in pixels) per label once
            label_px_sizes = []  # (w_px, h_px)
            temp_texts = []
            for i in range(Y.shape[0]):
                lab_txt = labels[i] if i < len(labels) else f"S{i+1}"
                tk = dict(text_kwargs_common)
                tk.setdefault("fontsize", label_sizes[i])
                tk.setdefault("weight", label_weight)
                # Keep artist renderable so get_window_extent works; make it invisible via alpha
                t = ax.text(0, 0, str(lab_txt), ha="center", va="center", alpha=0.0, **tk)
                temp_texts.append(t)
                bb = t.get_window_extent(renderer=renderer)
                label_px_sizes.append((float(bb.width), float(bb.height)))
            # Remove temp texts
            for t in temp_texts:
                try:
                    t.remove()
                except Exception:
                    pass

            # Helper to convert pixel width/height to data-rect centered at (x,y)
            inv = ax.transData.inverted()
            def rect_from_center_in_data(xc: float, yc: float, w_px: float, h_px: float):
                # Map center to pixels
                px_center = ax.transData.transform((xc, yc))
                half_w = 0.5 * w_px
                half_h = 0.5 * h_px
                ll = inv.transform((px_center[0] - half_w, px_center[1] - half_h))
                ur = inv.transform((px_center[0] + half_w, px_center[1] + half_h))
                x0, y0 = float(min(ll[0], ur[0])), float(min(ll[1], ur[1]))
                x1, y1 = float(max(ll[0], ur[0])), float(max(ll[1], ur[1]))
                return x0, y0, x1, y1

            def label_rect_from_anchor(xc: float, yc: float, w_px: float, h_px: float):
                # Convert requested anchor to a center so that text can be placed
                # while returning the rectangle in data coordinates.
                px_center = ax.transData.transform((xc, yc))
                ax_map = {
                    "center": (0.0, 0.0),
                    "middle": (0.0, 0.0),
                    "middle_middle": (0.0, 0.0),  # alias for center
                    "left": (0.5, 0.0),
                    "right": (-0.5, 0.0),
                    "top": (0.0, -0.5),
                    "bottom": (0.0, 0.5),
                    "top_left": (0.5, -0.5),
                    "top_right": (-0.5, -0.5),
                    "bottom_left": (0.5, 0.5),
                    "bottom_right": (-0.5, 0.5),
                    "middle_left": (0.5, 0.0),
                    "middle_right": (-0.5, 0.0),
                }
                dxs, dys = ax_map.get(label_anchor, (0.0, 0.0))
                px_center = (px_center[0] + dxs * w_px, px_center[1] + dys * h_px)
                ll = inv.transform((px_center[0] - 0.5 * w_px, px_center[1] - 0.5 * h_px))
                ur = inv.transform((px_center[0] + 0.5 * w_px, px_center[1] + 0.5 * h_px))
                x0, y0 = float(min(ll[0], ur[0])), float(min(ll[1], ur[1]))
                x1, y1 = float(max(ll[0], ur[0])), float(max(ll[1], ur[1]))
                return x0, y0, x1, y1

            # Build candidates per layer: sample across support where thickness>0
            # Orientation/anchor point must lie on the mid-line of the stream.
            # We therefore fix y at the mid-line (0.5 * thickness) and only vary x.
            y_fracs = (0.5,)  # mid-line only
            max_candidates_per_layer = int(max(1, label_balanced_candidates_per_layer))
            candidates_per_layer = []  # list[list[dict]]
            rects_per_layer = []       # list[list[tuple(x0,y0,x1,y1)]]

            # Current x-limits define the allowed plotting span. Keep a small
            # inset so labels do not touch the axes margins.
            xmin_lim, xmax_lim = ax.get_xlim()
            xr_lim = max(1e-9, (xmax_lim - xmin_lim))
            inset = max(0.0, float(label_balanced_inset_frac))
            # Effective usable span for candidates
            xmin_eff = xmin_lim + inset * xr_lim
            xmax_eff = xmax_lim - inset * xr_lim
            x_eps = inset * xr_lim
            # Convert pixel tolerance to data-units in y for vertical-fit checks
            y_eps_data = abs(inv.transform((0.0, float(label_balanced_fit_tolerance_px)))[1] - inv.transform((0.0, 0.0))[1])

            for i in range(Y.shape[0]):
                lab = labels[i] if i < len(labels) else f"S{i+1}"
                thickness = tops[i] - bottoms[i]
                mask = thickness > 0
                if not np.any(mask):
                    candidates_per_layer.append([])
                    rects_per_layer.append([])
                    continue

                # Find contiguous segments of True in mask
                idx = np.arange(len(Xp))
                segments = []
                in_seg = False
                seg_start = 0
                for j, m in enumerate(mask):
                    if m and not in_seg:
                        in_seg = True
                        seg_start = j
                    elif not m and in_seg:
                        in_seg = False
                        segments.append((seg_start, j - 1))
                if in_seg:
                    segments.append((seg_start, len(Xp) - 1))

                # We'll plot debug segments after thickness filtering so that
                # they reflect the actual candidate regions.

                # Distribute candidates roughly evenly across all segments
                total_len = sum(max(1, (b - a + 1)) for a, b in segments)
                num_target = min(max_candidates_per_layer, total_len)
                # Guard
                if num_target <= 0:
                    candidates_per_layer.append([])
                    rects_per_layer.append([])
                    continue

                # Compute thickness threshold for top-quantile filtering
                try:
                    q = float(label_balanced_min_thickness_q)
                except Exception:
                    q = 0.5
                q = min(1.0, max(0.0, q))
                thick_vals = thickness[mask]
                if thick_vals.size > 0:
                    thick_thresh = float(np.quantile(thick_vals, q))
                else:
                    thick_thresh = 0.0

                layer_candidates = []
                layer_rects = []
                filtered_segments_for_debug = []
                for a, b in segments:
                    seg_len = max(1, (b - a + 1))
                    # Number of samples from this segment proportional to length
                    seg_n = max(1, int(round(num_target * (seg_len / max(1, total_len)))))
                    # Evenly spaced indices in [a, b]
                    if seg_n >= seg_len:
                        sample_js = list(range(a, b + 1))
                    else:
                        sample_js = list(np.linspace(a, b, seg_n, dtype=int))
                    used = set()
                    # Per-segment thickness threshold (handles many short segments
                    # common with sorted stream order)
                    seg_thick = thickness[a:b+1]
                    seg_mask_pos = seg_thick > 0
                    if np.any(seg_mask_pos):
                        seg_thresh = float(np.quantile(seg_thick[seg_mask_pos], q))
                    else:
                        seg_thresh = 0.0
                    seg_had_candidate = False
                    # Build filtered subsegments for debug: contiguous js with thickness >= seg_thresh
                    if label_balanced_debug_segments:
                        run_start = None
                        for jj in range(a, b + 1):
                            if thickness[jj] >= seg_thresh:
                                if run_start is None:
                                    run_start = jj
                            else:
                                if run_start is not None:
                                    filtered_segments_for_debug.append((run_start, jj - 1))
                                    run_start = None
                        if run_start is not None:
                            filtered_segments_for_debug.append((run_start, b))
                    for j in sample_js:
                        if j in used:
                            continue
                        used.add(j)
                        # Skip positions below segment-level thickness quantile
                        if thickness[j] < seg_thresh:
                            continue
                        w_px, h_px = label_px_sizes[i]
                        for yf in y_fracs:
                            x = float(Xp[j])
                            y = float(bottoms[i, j] + yf * thickness[j])
                            rect = label_rect_from_anchor(x, y, w_px, h_px)
                            # Filter out candidates whose label box would leave x-limits
                            rx0, ry0, rx1, ry1 = rect
                            if (rx0 < xmin_eff + x_eps) or (rx1 > xmax_eff - x_eps):
                                # try to nudge x into bounds
                                if rx0 < xmin_eff + x_eps:
                                    dx = (xmin_eff + x_eps) - rx0
                                    x = x + dx
                                    rect = label_rect_from_anchor(x, y, w_px, h_px)
                                    rx0, ry0, rx1, ry1 = rect
                                if rx1 > xmax_eff - x_eps:
                                    dx = rx1 - (xmax_eff - x_eps)
                                    x = x - dx
                                    rect = label_rect_from_anchor(x, y, w_px, h_px)
                                    rx0, ry0, rx1, ry1 = rect
                                # if still out of bounds skip
                                if (rx0 < xmin_eff + x_eps) or (rx1 > xmax_eff - x_eps):
                                    continue
                            layer_candidates.append({"i": i, "x": x, "y": y, "align": label_anchor})
                            layer_rects.append(rect)
                            seg_had_candidate = True
                    # Ensure at least one position from this segment if all were filtered
                    if not seg_had_candidate:
                        j_peak_seg = int(a + np.argmax(seg_thick))
                        w_px, h_px = label_px_sizes[i]
                        x = float(Xp[j_peak_seg])
                        y = float(bottoms[i, j_peak_seg] + 0.5 * thickness[j_peak_seg])
                        rect = label_rect_from_anchor(x, y, w_px, h_px)
                        rx0, ry0, rx1, ry1 = rect
                        if (rx0 < xmin_eff + x_eps):
                            x += (xmin_eff + x_eps - rx0)
                            rect = label_rect_from_anchor(x, y, w_px, h_px)
                        if (rx1 > xmax_eff - x_eps):
                            x -= (rx1 - (xmax_eff - x_eps))
                            rect = label_rect_from_anchor(x, y, w_px, h_px)
                        layer_candidates.append({"i": i, "x": x, "y": y, "align": label_anchor})
                        layer_rects.append(rect)
                # Ensure at least one candidate per layer so we never drop labels
                if len(layer_candidates) == 0:
                    # fallback near fattest point
                    if np.any(mask):
                        j_peak = int(np.argmax(thickness))
                        w_px, h_px = label_px_sizes[i]
                        x = float(Xp[j_peak])
                        y = float(bottoms[i, j_peak] + 0.5 * thickness[j_peak])
                        rect = label_rect_from_anchor(x, y, w_px, h_px)
                        rx0, ry0, rx1, ry1 = rect
                        if (rx0 < xmin_eff + x_eps):
                            x += (xmin_eff + x_eps - rx0)
                            rect = label_rect_from_anchor(x, y, w_px, h_px)
                        if (rx1 > xmax_eff - x_eps):
                            x -= (rx1 - (xmax_eff - x_eps))
                            rect = label_rect_from_anchor(x, y, w_px, h_px)
                        layer_candidates.append({"i": i, "x": x, "y": y, "align": label_anchor})
                        layer_rects.append(rect)
                candidates_per_layer.append(layer_candidates)
                rects_per_layer.append(layer_rects)

                # Plot filtered segments used for candidates
                if label_balanced_debug_segments and filtered_segments_for_debug:
                    midline = bottoms[i] + 0.5 * thickness
                    dbg = {"color": "lime", "linewidth": 4.0, "alpha": 0.9}
                    if label_balanced_debug_kwargs:
                        dbg.update(dict(label_balanced_debug_kwargs))
                    for (a_f, b_f) in filtered_segments_for_debug:
                        xs = Xp[a_f:b_f+1]
                        ys = midline[a_f:b_f+1]
                        if len(xs) > 0:
                            ax.plot(xs, ys, **dbg)

            return candidates_per_layer, rects_per_layer, label_px_sizes

        def _rect_overlap_area(r0, r1) -> float:
            x0a, y0a, x1a, y1a = r0
            x0b, y0b, x1b, y1b = r1
            xi0 = max(x0a, x0b)
            yi0 = max(y0a, y0b)
            xi1 = min(x1a, x1b)
            yi1 = min(y1a, y1b)
            if xi1 <= xi0 or yi1 <= yi0:
                return 0.0
            return float((xi1 - xi0) * (yi1 - yi0))

        def _out_of_bounds_penalty(rect, xlim, ylim) -> float:
            x0, y0, x1, y1 = rect
            xmin, xmax = xlim
            ymin, ymax = ylim
            pen = 0.0
            if x0 < xmin:
                pen += (xmin - x0)
            if x1 > xmax:
                pen += (x1 - xmax)
            if y0 < ymin:
                pen += (ymin - y0)
            if y1 > ymax:
                pen += (y1 - ymax)
            return float(pen)

        def _sa_optimize(cands, rects, label_px_sizes, xlim, ylim):
            # Filter out empty layers
            layer_indices = [i for i, lst in enumerate(cands) if len(lst) > 0]
            if not layer_indices:
                return []

            # For layers with candidates, start from the middle candidate
            current_choice = {}
            for idx in layer_indices:
                current_choice[idx] = len(cands[idx]) // 2

            # Pixel helpers
            trans = ax.transData.transform
            def rect_to_px(rect):
                x0, y0, x1, y1 = rect
                (px0, py0) = trans((x0, y0))
                (px1, py1) = trans((x1, y1))
                return (min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1))
            def overlap_area_px(r0, r1):
                a0, b0, a1, b1 = rect_to_px(r0)
                c0, d0, c1, d1 = rect_to_px(r1)
                xi0 = max(a0, c0)
                yi0 = max(b0, d0)
                xi1 = min(a1, c1)
                yi1 = min(b1, d1)
                if xi1 <= xi0 or yi1 <= yi0:
                    return 0.0
                return float((xi1 - xi0) * (yi1 - yi0))

            # Precompute per-candidate and pairwise terms to accelerate cost evaluations
            avg_area_px = float(np.mean([label_px_sizes[i][0] * label_px_sizes[i][1] for i in layer_indices])) if layer_indices else 1.0
            avg_area_px = max(1.0, avg_area_px)
            avg_w_px = float(np.mean([label_px_sizes[i][0] for i in layer_indices])) if layer_indices else 20.0
            min_sep_px = 0.6 * avg_w_px

            # Per-layer caches
            px_rects_map = {i: [rect_to_px(r) for r in rects[i]] for i in layer_indices}
            centers_px_x_map = {i: [0.5 * (rp[0] + rp[2]) for rp in px_rects_map[i]] for i in layer_indices}
            centers_x_data_map = {i: [cands[i][j]["x"] for j in range(len(cands[i]))] for i in layer_indices}
            bounds_pen_map = {i: [_out_of_bounds_penalty(rects[i][j], xlim, ylim) for j in range(len(rects[i]))] for i in layer_indices}

            # Visibility once per candidate
            vis_pen_map = {}
            for i in layer_indices:
                vp = []
                for j, rect in enumerate(rects[i]):
                    x0, y0, x1, y1 = rect
                    xs = np.linspace(x0, x1, 5)
                    js = np.clip(np.searchsorted(Xp, xs), 1, len(Xp) - 1)
                    pen = 0.0
                    for xi, jj in zip(xs, js):
                        xL, xR = Xp[jj - 1], Xp[jj]
                        t = 0.0 if xR == xL else float((xi - xL) / (xR - xL))
                        bnd = float((1 - t) * bottoms[i, jj - 1] + t * bottoms[i, jj])
                        top = float((1 - t) * tops[i, jj - 1] + t * tops[i, jj])
                        if y0 < bnd:
                            pen += (bnd - y0)
                        if y1 > top:
                            pen += (y1 - top)
                    vp.append(pen)
                vis_pen_map[i] = vp

            # Prefer thick regions cache
            max_thick = np.max(tops - bottoms, axis=1)
            width_pen_map = {}
            for i in layer_indices:
                wp = []
                for cx in centers_x_data_map[i]:
                    jj = int(np.clip(np.searchsorted(Xp, cx), 1, len(Xp) - 1))
                    xL, xR = Xp[jj - 1], Xp[jj]
                    t = 0.0 if xR == xL else float((cx - xL) / (xR - xL))
                    bnd = float((1 - t) * bottoms[i, jj - 1] + t * bottoms[i, jj])
                    top = float((1 - t) * tops[i, jj - 1] + t * tops[i, jj])
                    thick_here = max(0.0, top - bnd)
                    mt = max(1e-9, float(max_thick[i]))
                    wp.append(1.0 - (thick_here / mt))
                width_pen_map[i] = wp

            # Pairwise precomputations for overlap, closeness, and repulsion
            pair_keys = []
            for a_pos in range(len(layer_indices)):
                for b_pos in range(a_pos + 1, len(layer_indices)):
                    pair_keys.append((layer_indices[a_pos], layer_indices[b_pos]))

            overlap_map = {}
            close_map = {}
            repulse_map = {}
            for ia, ib in pair_keys:
                ra = px_rects_map[ia]
                rb = px_rects_map[ib]
                ca = centers_px_x_map[ia]
                cb = centers_px_x_map[ib]
                A = len(ra)
                B = len(rb)
                ov = np.zeros((A, B), dtype=float)
                cl = np.zeros((A, B), dtype=float)
                rp = np.zeros((A, B), dtype=float)
                for a in range(A):
                    ax0, ay0, ax1, ay1 = ra[a]
                    cx_a = 0.5 * (ax0 + ax1)
                    cy_a = 0.5 * (ay0 + ay1)
                    for b in range(B):
                        bx0, by0, bx1, by1 = rb[b]
                        # overlap area normalized by avg label area
                        xi0 = max(ax0, bx0)
                        yi0 = max(ay0, by0)
                        xi1 = min(ax1, bx1)
                        yi1 = min(ay1, by1)
                        if xi1 > xi0 and yi1 > yi0:
                            ov[a, b] = ((xi1 - xi0) * (yi1 - yi0)) / avg_area_px
                        # closeness in x
                        dx = abs(ca[a] - cb[b])
                        if dx < min_sep_px:
                            cl[a, b] = (min_sep_px - dx)
                        # repulsion heuristic based on center distances
                        cx_b = 0.5 * (bx0 + bx1)
                        cy_b = 0.5 * (by0 + by1)
                        wx = 0.5 * ((ax1 - ax0) + (bx1 - bx0))
                        hy = 0.6 * 0.5 * ((ay1 - ay0) + (by1 - by0))
                        dx2 = abs(cx_a - cx_b)
                        dy2 = abs(cy_a - cy_b)
                        if dx2 < wx and dy2 < hy:
                            rp[a, b] = (wx - dx2) * (hy - dy2)
                overlap_map[(ia, ib)] = ov
                close_map[(ia, ib)] = cl
                repulse_map[(ia, ib)] = rp

            def total_cost(choice):
                # Gather center positions and fast per-candidate costs
                centers_px = []
                bounds = 0.0
                vis = 0.0
                widthp = 0.0
                for i in layer_indices:
                    sel = choice[i]
                    centers_px.append(centers_px_x_map[i][sel])
                    bounds += bounds_pen_map[i][sel]
                    vis += vis_pen_map[i][sel]
                    widthp += width_pen_map[i][sel]

                # Pairwise contributions
                overlap = 0.0
                close_pen = 0.0
                repulse_pen = 0.0
                for a_pos in range(len(layer_indices)):
                    ia = layer_indices[a_pos]
                    for b_pos in range(a_pos + 1, len(layer_indices)):
                        ib = layer_indices[b_pos]
                        sa = choice[ia]
                        sb = choice[ib]
                        overlap += overlap_map[(ia, ib)][sa, sb]
                        close_pen += close_map[(ia, ib)][sa, sb]
                        repulse_pen += repulse_map[(ia, ib)][sa, sb]

                # Grid penalty depends only on sorted x-centers
                x0_px = trans((xlim[0], ylim[0]))[0]
                x1_px = trans((xlim[1], ylim[0]))[0]
                L = len(centers_px)
                inset_px = float(max(0.0, label_balanced_inset_frac)) * (x1_px - x0_px)
                left = x0_px + inset_px
                right = x1_px - inset_px
                grid_px = np.linspace(left, right, max(L, 1))
                span_px = max(1.0, (right - left))
                centers_px_sorted = sorted(centers_px)
                grid_pen = 0.0
                for r in range(L):
                    grid_pen += ((centers_px_sorted[r] - grid_px[r]) / span_px) ** 2

                hard_overlap_pen = 12.0 * overlap
                return hard_overlap_pen + 12.0 * bounds + 0.9 * close_pen + 0.9 * grid_pen + 0.8 * repulse_pen + 6.0 * vis + 0.8 * widthp

            best_choice = dict(current_choice)
            best_cost = total_cost(current_choice)

            # Simple simulated annealing (with optional tqdm over temperature steps)
            import math, random
            T0 = float(label_balanced_T0)
            T_min = float(label_balanced_T_min)
            alpha = float(label_balanced_alpha)
            iters_per_T = int(max(1, label_balanced_iters_per_T))

            def run_one_sa(start_choice: dict) -> tuple[dict, float]:
                cur = dict(start_choice)
                best = dict(start_choice)
                best_c = total_cost(cur)
                # compute number of temperature steps
                num_steps = max(1, int(math.ceil(math.log(max(T_min, 1e-9) / max(T0, 1e-9)) / math.log(max(alpha, 1e-9)))))
                T = T0
                step_iter = range(num_steps)
                if use_tqdm and _tqdm_fn is not None:
                    step_iter = _tqdm_fn(step_iter, desc="Balancing labels", leave=False)
                for _ in step_iter:
                    for _ in range(iters_per_T):
                        idx = random.choice(layer_indices)
                        if len(cands[idx]) <= 1:
                            continue
                        cur_idx = cur[idx]
                        proposal = cur_idx
                        tries = 0
                        while proposal == cur_idx and tries < 8:
                            proposal = random.randrange(0, len(cands[idx]))
                            tries += 1
                        if proposal == cur_idx:
                            continue
                        trial = dict(cur)
                        trial[idx] = proposal
                        c_new = total_cost(trial)
                        c_cur = total_cost(cur)
                        accept = (c_new <= c_cur) or (math.exp(-(c_new - c_cur) / max(T, 1e-9)) > random.random())
                        if accept:
                            cur = trial
                            if c_new < best_c:
                                best_c = c_new
                                best = dict(trial)
                    T *= alpha
                return best, best_c

            # Random restarts
            best_choice = dict(current_choice)
            best_cost = total_cost(current_choice)
            restarts = int(max(1, label_balanced_restarts))
            for _ in range(restarts):
                # start from a random initial choice
                start = {idx: (0 if len(cands[idx]) == 0 else random.randrange(0, len(cands[idx]))) for idx in layer_indices}
                cand_choice, cand_cost = run_one_sa(start)
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_choice = cand_choice

            # Build final targets list
            out = []
            for idx in layer_indices:
                pick = best_choice[idx]
                cand = cands[idx][pick]
                out.append((cand["i"], cand["x"], cand["y"], cand["align"]))
            return out

        for i in range(Y.shape[0]):
            thickness = tops[i] - bottoms[i]
            if np.all(thickness <= 0):
                continue

            if position == "peak":
                j = int(np.argmax(thickness))
                x = Xp[j]
                y = bottoms[i, j] + 0.5 * thickness[j]
                label_targets.append((i, x, y, "center"))
            elif position == "max_width":
                if thickness[j_wide] <= 0:
                    # skip labels that are not present at the widest slice
                    continue
                x = Xp[j_wide]
                y = bottoms[i, j_wide] + 0.5 * thickness[j_wide]
                label_targets.append((i, x, y, "center"))
            elif position == "start":
                nz = np.where(thickness > 0)[0]
                if nz.size == 0:
                    continue
                j0 = int(nz[0])
                x = Xp[j0] - x_margin
                y = bottoms[i, j0] + 0.5 * thickness[j0]
                label_targets.append((i, x, y, "right"))
            elif position == "end":
                nz = np.where(thickness > 0)[0]
                if nz.size == 0:
                    continue
                j1 = int(nz[-1])
                x = Xp[j1] + x_margin
                y = bottoms[i, j1] + 0.5 * thickness[j1]
                label_targets.append((i, x, y, "left"))
            elif position == "sliding_window":
                # Placed later after defining helper; handled after loop
                pass
            else:  # position == 'balanced'
                # Build and optimize candidates only once, then break loop to render
                pass

        if position == "balanced":
            # Build candidates and run optimizer
            cands, rects, label_px_sizes = _balanced_candidates_and_text_extents()
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            optimized = _sa_optimize(cands, rects, label_px_sizes, (xmin, xmax), (ymin, ymax))
            # Replace label_targets with optimized
            label_targets = optimized
        elif position == "sliding_window":
            # Fast in-stream labeler per Di Bartolomeo & Hu (2016)
            for i in range(Y.shape[0]):
                lab = labels[i] if i < len(labels) else f"S{i+1}"
                color_i = (label_color if isinstance(label_color, str) else (label_color[i] if label_color else "black"))
                try:
                    _place_ok = place_label_fast_on_layer(
                        ax, Xp, bottoms[i], tops[i], str(lab),
                        min_fontsize=8, max_fontsize=int(label_sizes[i]),
                        fontfamily=None, fontweight=label_weight,
                        color=color_i,
                    )
                except Exception:
                    # Fallback: place at peak like 'peak' mode
                    thickness = tops[i] - bottoms[i]
                    if np.all(thickness <= 0):
                        continue
                    j = int(np.argmax(thickness))
                    x = Xp[j]
                    y = bottoms[i, j] + 0.5 * thickness[j]
                    ax.text(x, y, str(lab), ha="center", va="center",
                            fontsize=label_sizes[i], weight=label_weight, color=color_i)

        # De-overlap start/end labels vertically using a greedy pass
        if position in {"start", "end"} and label_targets:
            ymin0, ymax0 = ax.get_ylim()
            y_range = ymax0 - ymin0 if ymax0 > ymin0 else 1.0
            min_gap = max(label_min_gap_frac, 0.0) * y_range
            order = sorted(range(len(label_targets)), key=lambda idx: label_targets[idx][2])
            adjusted = list(label_targets)
            for pos_idx, idx in enumerate(order):
                if pos_idx == 0:
                    continue
                prev_idx = order[pos_idx - 1]
                i_prev, x_prev, y_prev, a_prev = adjusted[prev_idx]
                i_cur, x_cur, y_cur, a_cur = adjusted[idx]
                if y_cur < y_prev + min_gap:
                    y_cur = y_prev + min_gap
                    y_cur = min(y_cur, ymax0 - 0.5 * min_gap)
                    adjusted[idx] = (i_cur, x_cur, y_cur, a_cur)
            label_targets = adjusted

        # Render labels
        def _anchor_to_aligns(anchor: str) -> Tuple[str, str]:
            m = {
                "center": ("center", "center"),
                "left": ("left", "center"),
                "right": ("right", "center"),
                "middle_left": ("left", "center"),
                "middle_right": ("right", "center"),
                "top": ("center", "top"),
                "bottom": ("center", "bottom"),
                "top_left": ("left", "top"),
                "top_right": ("right", "top"),
                "bottom_left": ("left", "bottom"),
                "bottom_right": ("right", "bottom"),
            }
            return m.get(anchor, ("center", "center"))

        anchor_points = []
        for i, x, y, align in label_targets:
            lab_txt = labels[i] if i < len(labels) else f"S{i+1}"
            tk = dict(text_kwargs_common)
            tk.setdefault("fontsize", label_sizes[i])
            tk.setdefault("weight", label_weight)
            tk.setdefault("color", label_colors[i])
            if position in {"peak", "max_width", "balanced"}:
                # For balanced, snap the anchor point to the stream mid-line at x
                if position == "balanced":
                    jn = int(np.clip(np.searchsorted(Xp, x), 1, len(Xp) - 1))
                    xL, xR = Xp[jn - 1], Xp[jn]
                    t = 0.0 if xR == xL else float((x - xL) / (xR - xL))
                    bnd = float((1 - t) * bottoms[i, jn - 1] + t * bottoms[i, jn])
                    top = float((1 - t) * tops[i, jn - 1] + t * tops[i, jn])
                    y = bnd + 0.5 * (top - bnd)
                ha, va = _anchor_to_aligns(label_anchor)
                ax.text(x, y, str(lab_txt), ha=ha, va=va, **tk)
                anchor_points.append((x, y))
            else:
                ax.text(x, y, str(lab_txt), ha=align, va="center", **tk)

        # Optional connectors for start/end
        if position in {"start", "end"} and label_connectors and label_targets:
            for i, x, y, align in label_targets:
                thickness = tops[i] - bottoms[i]
                nz = np.where(thickness > 0)[0]
                if nz.size == 0:
                    continue
                j_edge = int(nz[0] if position == "start" else nz[-1])
                x_edge = Xp[j_edge]
                y_edge = bottoms[i, j_edge] + 0.5 * thickness[j_edge]
                color = label_connector_color or label_colors[i] or "#666"
                ax.plot([x, x_edge], [y, y_edge], color=color,
                        alpha=label_connector_alpha, linewidth=label_connector_linewidth)

        # Optional anchor point plotting
        if label_plot_anchors and anchor_points:
            pk = dict(marker='o', linestyle='None', markersize=20, color='k', alpha=0.5)
            if label_point_kwargs:
                pk.update(label_point_kwargs)
            xs, ys = zip(*anchor_points)
            ax.plot(xs, ys, **pk)

        # Extend x-limits if needed so labels are not clipped at edges
        if extend_left or extend_right:
            xmin, xmax = ax.get_xlim()
            if extend_left:
                xmin = xmin - 2 * x_margin
            if extend_right:
                xmax = xmax + 2 * x_margin
            ax.set_xlim(xmin, xmax)

    return ax


# ---------- Global ordering (BestFirst + TwoOpt) per Di Bartolomeo & Hu (2016) ----------

def _l1_wiggle_of_midline(X: np.ndarray, mid: np.ndarray) -> float:
    """Sum of absolute slope of the midline, discrete 1-norm."""
    g = np.gradient(mid, X)
    return float(np.sum(np.abs(g)))


def _wiggle_increment_when_added(X: np.ndarray, side_baseline: np.ndarray, y: np.ndarray) -> float:
    """Wiggle increment if we add layer y on this side (midline = base + 0.5*y)."""
    return _l1_wiggle_of_midline(X, side_baseline + 0.5 * y)


def order_bestfirst(X: np.ndarray, Y: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Return (center_out_top, center_out_bottom) lists of layer indices.
    Greedy bipartition that adds the layer/side with lowest incremental L1 midline wiggle.

    Reference: Di Bartolomeo & Hu (2016), BestFirst stage.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    k, n = Y.shape
    remaining = list(range(k))
    top_base = np.zeros(n, float)
    bot_base = np.zeros(n, float)
    top_list: list[int] = []
    bot_list: list[int] = []
    while remaining:
        best: Optional[tuple[float, str, int]] = None
        for i in remaining:
            y = Y[i]
            s_top = _wiggle_increment_when_added(X, top_base, y)
            s_bot = _wiggle_increment_when_added(X, bot_base, y)
            if best is None or s_top < best[0]:
                best = (s_top, "top", i)
            if s_bot < best[0]:
                best = (s_bot, "bot", i)
        _, side, pick = best  # type: ignore
        remaining.remove(pick)
        if side == "top":
            top_list.append(pick)
            top_base = top_base + Y[pick]
        else:
            bot_list.append(pick)
            bot_base = bot_base + Y[pick]
    return top_list, bot_list


def _inside_out_scan(X: np.ndarray, side_list: list[int], Y: np.ndarray) -> list[int]:
    """
    Rebuild one side inside-out: choose head or tail, whichever lowers incremental wiggle.
    """
    from collections import deque
    dq = deque(side_list)
    n = Y.shape[1]
    base = np.zeros(n, float)
    out: list[int] = []
    while dq:
        cand_left = dq[0]
        cand_right = dq[-1] if len(dq) > 1 else dq[0]
        sL = _wiggle_increment_when_added(X, base, Y[cand_left])
        sR = _wiggle_increment_when_added(X, base, Y[cand_right])
        if sL <= sR:
            pick = dq.popleft()
        else:
            pick = dq.pop()
        out.append(pick)
        base = base + Y[pick]
    return out


def order_twoopt(
    X: np.ndarray,
    Y: np.ndarray,
    repeats: int = 8,
    scans: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> list[int]:
    """
    Build BestFirst bipartition, then improve each side with inside-out scans,
    repeating from shuffled starts to avoid bad bipartitions. Returns a global order.

    Reference: Di Bartolomeo & Hu (2016), TwoOpt stage.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    best_order: Optional[list[int]] = None
    best_score = np.inf

    base_top, base_bot = order_bestfirst(X, Y)

    for _ in range(max(1, repeats)):
        top = base_top.copy()
        bot = base_bot.copy()
        rng.shuffle(top)
        rng.shuffle(bot)

        for _s in range(max(1, scans)):
            top = _inside_out_scan(X, top, Y)
            bot = _inside_out_scan(X, bot, Y)

        # Interleave bottom and top (center-out)
        interleaved: list[int] = []
        for a, b in zip(bot, top):
            interleaved.append(a)
            interleaved.append(b)
        if len(bot) > len(top):
            interleaved.extend(bot[len(top):])
        elif len(top) > len(bot):
            interleaved.extend(top[len(bot):])

        # Evaluate total L1 midline wiggle when stacking center-out
        n = Y.shape[1]
        g0 = np.zeros(n, float)
        score = 0.0
        for j in interleaved:
            mid = g0 + 0.5 * Y[j]
            score += _l1_wiggle_of_midline(X, mid)
            g0 = g0 + Y[j]
        if score < best_score:
            best_score = score
            best_order = interleaved

    return best_order if best_order is not None else list(range(Y.shape[0]))


# ---------- Fast in-stream labeler (sliding-window) per Di Bartolomeo & Hu (2016) ----------

from collections import deque


def _sliding_window_min(a: np.ndarray, w: int) -> np.ndarray:
    """Return sliding-window minimum over a 1D array using a deque.

    Parameters
    ----------
    a : ndarray
        Input array of length m.
    w : int
        Window size (1 <= w <= m).

    Returns
    -------
    ndarray, shape (m - w + 1,)
        The minimum for each contiguous window of width ``w``.
    """
    a = np.asarray(a, float)
    m = len(a)
    if w <= 0 or w > m:
        raise ValueError("bad window")
    dq = deque()
    out = np.empty(m - w + 1, float)
    for i, val in enumerate(a):
        while dq and dq[0] <= i - w:
            dq.popleft()
        while dq and a[dq[-1]] >= val:
            dq.pop()
        dq.append(i)
        if i >= w - 1:
            out[i - w + 1] = a[dq[0]]
    return out


def _sliding_window_max(a: np.ndarray, w: int) -> np.ndarray:
    """Return sliding-window maximum over a 1D array using a deque.

    Parameters
    ----------
    a : ndarray
        Input array of length m.
    w : int
        Window size (1 <= w <= m).

    Returns
    -------
    ndarray, shape (m - w + 1,)
        The maximum for each contiguous window of width ``w``.
    """
    return -_sliding_window_min(-np.asarray(a, float), w)


def _label_window_search(
    top: np.ndarray,
    bot: np.ndarray,
    w: int,
    sigma: float,
) -> Optional[tuple[int, float]]:
    """
    Return (i, hmax) where i is the left index of the best window and hmax is Ti-Bi.
    Windows are [i, i+w-1]. If no feasible window, return None.
    """
    T = _sliding_window_min(top, w)
    B = _sliding_window_max(bot, w)
    H = T - B
    imax = int(np.argmax(H))
    hmax = float(H[imax])
    if hmax > 0 and (w / hmax) <= sigma:
        return imax, hmax
    return None


def place_label_fast_on_layer(
    ax: plt.Axes,
    Xp: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    text: str,
    min_fontsize: int = 6,
    max_fontsize: int = 18,
    fontfamily: Optional[str] = None,
    fontweight: str = "bold",
    shrink: float = 0.9,
    color: str = "black",
    sigma_override: Optional[float] = None,
) -> bool:
    """
    Try to place `text` inside the stream using the paper's O(m log W) scheme.
    Returns True if placed.
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure renderer exists
    renderer = fig.canvas.get_renderer()

    # Average pixel per sample along X
    px0, _ = ax.transData.transform((float(Xp[0]), 0.0))
    px1, _ = ax.transData.transform((float(Xp[-1]), 0.0))
    avg_px_per_sample = max(1.0, abs(px1 - px0) / max(1, len(Xp) - 1))

    def text_px_wh(fontsize: float) -> tuple[float, float]:
        t = ax.text(0, 0, text, fontsize=fontsize, fontfamily=fontfamily, weight=fontweight, alpha=0.0)
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        return float(bb.width), float(bb.height)

    fs = float(max_fontsize)
    while fs >= float(min_fontsize):
        w_px, h_px = text_px_wh(fs)
        w_samples = int(max(1, round(w_px / avg_px_per_sample)))
        (x_mid, y0) = ax.transData.inverted().transform((px0, 0.0))
        (x_mid, y1) = ax.transData.inverted().transform((px0, h_px))
        h_data = abs(y1 - y0)
        sigma = sigma_override if sigma_override is not None else (w_samples / max(h_data, 1e-9))

        try:
            found = _label_window_search(top, bottom, w_samples, sigma)
        except ValueError:
            found = None
        if found is not None:
            i_left, hmax = found
            j_center = i_left + w_samples // 2
            j_center = int(np.clip(j_center, 0, len(Xp) - 1))
            x = float(Xp[j_center])
            y = 0.5 * float(top[j_center] + bottom[j_center])
            ax.text(x, y, text,
                    fontsize=fs, fontfamily=fontfamily, weight=fontweight,
                    color=color, ha="center", va="center", clip_on=True)
            return True
        fs = max(min_fontsize, fs * shrink)
        if fs == min_fontsize:
            fs -= 1
    return False


# ---------- Minimal demo when executed directly ----------

def _demo():
    """Minimal demonstration of the streamgraph with linear and smoothed edges."""
    rng = np.random.default_rng(7)
    n, k = 40, 5
    X = np.arange(n)
    base = np.linspace(0, 2*np.pi, n)
    Y = []
    for i in range(k):
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.6, 1.3)
        y = amp * (np.sin(base + phase) + 1.2) + rng.normal(0, 0.08, size=n) + 0.15
        y = np.clip(y, 0, None)
        Y.append(y)
    Y = np.vstack(Y)
    labels = list("ABCDE")

    # Figure 1: straight edges
    ax1 = plot_streamgraph(X, Y, labels=labels, sorted_streams=True,
                           margin_frac=0.10, smooth_window=1, cmap=None,
                           curve_samples=1)
    ax1.set_title("Streamgraph with linear boundaries")
    plt.show()

    # Figure 2: spline-smoothed boundaries
    ax2 = plot_streamgraph(X, Y, labels=labels, sorted_streams=True,
                           margin_frac=0.10, smooth_window=1, cmap=None,
                           curve_samples=16)
    ax2.set_title("Streamgraph with Catmull–Rom boundaries")
    plt.show()


if __name__ == "__main__":
    _demo()


