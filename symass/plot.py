"""
Nuclear chart plotting utilities.

Public functions
----------------
plot_map          -- single-column N-Z colour map
plot_maps         -- multi-column grid of N-Z colour maps
plot_predictions  -- true-vs-predicted + optional residual panels (train & test)
plot_shap         -- SHAP feature-importance bar chart + beeswarm summary plot
plot_correlations -- feature correlation heatmap + optional scatter matrix
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .constants import SHELLS_HO, SHELLS_MAGIC  # noqa: F401  (re-exported for convenience)

_SHELL_LINE_COLOR  = "#888888"   # medium gray — visible on both white bg and scatter points
_SHELL_LINE_LW     = 0.7
_SHELL_LINE_LS     = ":"

_CHART_SCALE = 12   # inches per 1 unit of N or Z range (1/12 inch per nucleon)

# ── Typography ─────────────────────────────────────────────────────────────────
# All font sizes live here so the whole library is tuned from one place.
_FS_TITLE   = 12   # panel / figure titles
_FS_LABEL   = 11   # axis labels (xlabel / ylabel)
_FS_TICK    = 10   # tick labels
_FS_LEGEND  = 10   # legend text
_FS_ANN     = 9    # in-cell annotations (correlation matrix numbers)
_FS_SMALL   = 9    # colourbar labels, suptitles, secondary text

# ── Column-name → LaTeX display string ────────────────────────────────────────
_LATEX_NAMES: dict[str, str] = {
    # SU(4) Casimir invariants
    "C2_4":        r"$\mathcal{C}_2[\mathrm{SU}(4)]$",
    "C3_4":        r"$\mathcal{C}_3[\mathrm{SU}(4)]$",
    "C3_4_pos":    r"$\mathcal{C}_3^{+}[\mathrm{SU}(4)]$",
    "C3_4_neg":    r"$\mathcal{C}_3^{-}[\mathrm{SU}(4)]$",
    "C4_4":        r"$\mathcal{C}_4[\mathrm{SU}(4)]$",
    # SU(3) Casimir invariants
    "C2_3":        r"$\mathcal{C}_2[\mathrm{SU}(3)]$",
    "C3_3":        r"$\mathcal{C}_3[\mathrm{SU}(3)]$",
    # SU(3) irrep labels
    "lam":         r"$\lambda$",
    "mu":          r"$\mu$",
    # HO quantum numbers
    "Nhw":         r"$N_{\hbar\omega}$",
    "shell":       r"$\eta$",
    # SU(4) irrep labels
    "P1":          r"$P_1$",
    "P2":          r"$P_2$",
    "P3":          r"$P_3$",
    # Nuclear numbers
    "A":           r"$A$",
    "A_23":        r"$A^{2/3}$",
    "Z":           r"$Z$",
    "N":           r"$N$",
    "T_z":         r"$T_z$",
    # Binding energies
    "BE_total_MeV":  r"$BE_\mathrm{total}\ (\mathrm{MeV})$",
    "BE_per_A_MeV":  r"$BE/A\ (\mathrm{MeV})$",
    # MDN outputs
    "mu":            r"$\mu\ (\mathrm{MeV})$",
    "sigma":         r"$\sigma\ (\mathrm{MeV})$",
}


def _fmt(col: str) -> str:
    """Return the LaTeX display name for *col*, falling back to the raw string."""
    return _LATEX_NAMES.get(col, col)


def _auto_figsize(max_N: int, max_Z: int) -> tuple[float, float]:
    """Return a figure size (width, height) proportional to the nuclear chart extent."""
    return (max_N / _CHART_SCALE, max_Z / _CHART_SCALE)


def _inset_colorbar(fig, ax, sc) -> None:
    """
    Add a compact horizontal colorbar inset at the top-left of *ax*.

    Tick labels are in the range [0.1, 9.9] in the scaled space, with a shared
    ×10^n multiplier placed to the right of the bar.  The exponent is derived
    from ``sc.get_clim()`` max_abs so the scaling is consistent regardless of
    data magnitude.  Whole-number ticks are displayed without a decimal point
    (e.g. "2"); non-integer ticks show one decimal place (e.g. "1.5").
    """
    cax  = ax.inset_axes([0.08, 0.88, 0.45, 0.04])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")

    vmin, vmax = sc.get_clim()
    max_abs = max(abs(vmin), abs(vmax))

    if max_abs == 0 or not np.isfinite(max_abs):
        cax.tick_params(labelsize=_FS_TICK, length=3, pad=2)
        return

    # Exponent that maps the largest value into the [1, 10) range
    exp   = int(np.floor(np.log10(max_abs)))
    scale = 10.0 ** exp

    # Generate ≤5 ticks in the scaled space.
    # No integer=True — allows steps of 0.5, 0.2, 0.1 when the range is
    # narrow, so labels can be anywhere in [0.1, 9.9] rather than 1–9 only.
    locator = mticker.MaxNLocator(nbins=5, prune="both")
    ticks_s = locator.tick_values(vmin / scale, vmax / scale)
    ticks   = ticks_s * scale

    # Clamp strictly to the colour range (floating-point tolerance)
    eps   = 1e-9 * max_abs
    ticks = ticks[(ticks >= vmin - eps) & (ticks <= vmax + eps)]

    if len(ticks):
        cbar.set_ticks(ticks)
        def _fmt_tick(x, _):
            v = x / scale
            # Show integer label when value is within 2 % of a whole number,
            # otherwise one decimal place (covers 0.5, 1.5, 2.5, … cleanly).
            return f"{round(v):.0f}" if abs(v - round(v)) < 0.02 * (abs(v) + 1e-12) else f"{v:.1f}"
        cbar.formatter = mticker.FuncFormatter(_fmt_tick)
        cbar.update_ticks()

    # Multiplier label — omit when exp == 0 (×10⁰ = ×1 is redundant)
    if exp != 0:
        cax.text(1.03, 0.5, rf"$\times\!10^{{{exp}}}$",
                 transform=cax.transAxes, ha="left", va="center",
                 fontsize=_FS_SMALL, clip_on=False)

    cax.tick_params(labelsize=_FS_TICK, length=3, pad=2)


def _apply_scale(vals: np.ndarray, scale: str) -> tuple[np.ndarray, mcolors.Normalize | None]:
    """
    Return (display_values, norm) for a given scaling mode.

    scale : "linear"   -- no transformation, matplotlib handles the colour range
            "log"      -- symmetric log (SymLogNorm), safe for signed values
            "norm"     -- min-max normalise to [0, 1] before plotting
    """
    if scale == "log":
        # SymLogNorm handles negatives; linthresh sets the linear region near zero.
        # Filter out zeros AND non-finite values before computing the percentile so
        # we never pass an empty array to nanpercentile.
        nz = vals[np.isfinite(vals) & (vals != 0)]
        linthresh = max(1.0, float(np.percentile(np.abs(nz), 10))) if len(nz) else 1.0
        norm = mcolors.SymLogNorm(
            linthresh=linthresh,
            vmin=float(np.nanmin(vals)),
            vmax=float(np.nanmax(vals)),
        )
        return vals, norm

    if scale == "norm":
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        vals = (vals - vmin) / (vmax - vmin + 1e-12)
        return vals, None

    # linear (default)
    return vals, None


def plot_map(
    df: pd.DataFrame,
    col: str,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    scale: str = "linear",
    point_size: int = 10,
    shells: list[int] | None = None,
    title_out: bool = True,
    savefig: str | None = None,
    dpi: int = 500,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw a single N-Z colour map for *col*.

    Parameters
    ----------
    df         : DataFrame that must contain columns "N", "Z", and *col*.
    col        : Column to visualise (feature, target, or any numeric column).
    figsize    : Figure size in inches ``(width, height)``.  If None (default),
                 auto-sized proportionally to ``max(N)`` and ``max(Z)`` using
                 a scale of 1/12 inch per nucleon.
    cmap       : Matplotlib colormap name (e.g. "viridis", "RdBu_r", "plasma").
    scale      : Colour scaling — "linear" | "log" | "norm".
    point_size : Marker size for scatter points.
    shells     : Shell-closure values to draw as grid lines.
                 Defaults to SHELLS_HO. Pass [] to suppress lines,
                 or SHELLS_MAGIC for empirical magic numbers.
    title_out  : If True (default), the panel title appears above the axes in
                 the standard matplotlib position.  If False, the title is
                 placed as a compact text annotation in the bottom-right corner
                 *inside* the axes — saves vertical whitespace in multi-panel
                 figures intended for publication.

    Returns
    -------
    fig, ax
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    if shells is None:
        shells = SHELLS_HO

    N    = df["N"].to_numpy()
    Z    = df["Z"].to_numpy()
    vals = df[col].to_numpy(dtype=float)

    max_N, max_Z = int(N.max()), int(Z.max())
    if figsize is None:
        figsize = _auto_figsize(max_N, max_Z)

    display_vals, norm = _apply_scale(vals, scale)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(N, Z, c=display_vals, cmap=cmap, s=point_size,
                    linewidths=0, norm=norm)
    for m in shells:
        ax.axvline(m, color=_SHELL_LINE_COLOR, lw=_SHELL_LINE_LW, ls=_SHELL_LINE_LS)
        ax.axhline(m, color=_SHELL_LINE_COLOR, lw=_SHELL_LINE_LW, ls=_SHELL_LINE_LS)

    ax.set_xlim(0, max_N)
    ax.set_ylim(0, max_Z)
    ax.set_aspect("equal", adjustable="box")
    _inset_colorbar(fig, ax, sc)
    ax.set_xlabel("Neutron number $N$")
    ax.set_ylabel("Proton number $Z$")
    title = _fmt(col) + (f"  [{scale}]" if scale != "linear" else "")
    if title_out:
        ax.set_title(title, fontsize=_FS_TITLE)
    else:
        ax.text(0.98, 0.03, title,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_FS_TITLE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))
    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_maps(
    df: pd.DataFrame,
    cols: list[str],
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    scale: str = "linear",
    point_size: int = 8,
    shells: list[int] | None = None,
    title_out: bool = True,
    savefig: str | None = None,
    dpi: int = 500,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Draw a grid of N-Z colour maps, one panel per column in *cols*.

    Parameters
    ----------
    df         : DataFrame that must contain columns "N", "Z", and all *cols*.
    cols       : List of column names to visualise.
    ncols      : Number of panels per row (default 3).
    figsize    : Total figure size.  If None (default), each panel is sized
                 proportionally to ``max(N)`` and ``max(Z)`` (1/12 inch per
                 nucleon), and panels are tiled to fill the grid.
    cmap       : Matplotlib colormap name.
    scale      : Colour scaling applied to every panel — "linear" | "log" | "norm".
    point_size : Marker size for scatter points.
    shells     : Shell-closure values to draw as grid lines.
                 Defaults to SHELLS_HO. Pass [] to suppress lines,
                 or SHELLS_MAGIC for empirical magic numbers.
    title_out  : If True (default), each panel title appears above the axes.
                 If False, the title is placed as a compact text annotation in
                 the bottom-right corner inside the axes — saves vertical
                 whitespace in publication figures.

    Returns
    -------
    fig, axes  (axes is a 2-D ndarray of shape (nrows, ncols))
    """
    if shells is None:
        shells = SHELLS_HO

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    N = df["N"].to_numpy()
    Z = df["Z"].to_numpy()

    max_N, max_Z = int(N.max()), int(Z.max())
    n     = len(cols)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        pw, ph  = _auto_figsize(max_N, max_Z)   # per-panel size
        figsize = (pw * ncols, ph * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, col in enumerate(cols):
        row, c = divmod(idx, ncols)
        ax   = axes[row][c]
        vals = df[col].to_numpy(dtype=float)
        display_vals, norm = _apply_scale(vals, scale)

        sc = ax.scatter(N, Z, c=display_vals, cmap=cmap, s=point_size,
                        linewidths=0, norm=norm)
        for m in shells:
            ax.axvline(m, color=_SHELL_LINE_COLOR, lw=_SHELL_LINE_LW, ls=_SHELL_LINE_LS)
            ax.axhline(m, color=_SHELL_LINE_COLOR, lw=_SHELL_LINE_LW, ls=_SHELL_LINE_LS)

        ax.set_xlim(0, max_N)
        ax.set_ylim(0, max_Z)
        ax.set_aspect("equal", adjustable="box")
        _inset_colorbar(fig, ax, sc)
        ax.set_xlabel("Neutron number $N$")
        ax.set_ylabel("Proton number $Z$")
        title = _fmt(col) + (f"  [{scale}]" if scale != "linear" else "")
        if title_out:
            ax.set_title(title, fontsize=_FS_TITLE)
        else:
            ax.text(0.98, 0.03, title,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=_FS_TITLE,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, c = divmod(idx, ncols)
        axes[row][c].set_visible(False)

    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axes


# ── Prediction diagnostics ─────────────────────────────────────────────────────

def plot_predictions(
    tr_metrics: dict,
    te_metrics: dict,
    tr_N: np.ndarray,
    tr_Z: np.ndarray,
    te_N: np.ndarray,
    te_Z: np.ndarray,
    target_name: str = "BE_total",
    residuals: bool = True,
    true_vs_pred: bool = True,
    shells: list[int] | None = None,
    val_metrics: dict | None = None,
    val_N: np.ndarray | None = None,
    val_Z: np.ndarray | None = None,
    savefig: str | None = None,
    dpi: int = 500,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Diagnostic plots comparing true and predicted values for train and test sets.

    Parameters
    ----------
    tr_metrics, te_metrics : dicts returned by evaluate().  Must contain keys
                             ``"y_true"`` and ``"y_pred"`` as flat 1-D arrays.
    tr_N, tr_Z             : Neutron / proton numbers for training nuclei.
    te_N, te_Z             : Neutron / proton numbers for test nuclei.
    target_name            : Label used on axes (default ``"BE_total"``).
    residuals              : If ``True`` (default), add a residual heatmap on
                             the nuclear chart alongside the true-vs-predicted
                             scatter.  If ``False``, show only the scatter.
    shells                 : Shell-closure lines drawn on the residual chart.
                             Defaults to ``SHELLS_MAGIC``.  Pass ``[]`` to
                             suppress.
    val_metrics            : Optional evaluate() dict for the validation set
                             (AME2020 new nuclei).  When provided, validation
                             nuclei are overlaid on the residual heatmap with
                             black outlines so they stand out from the
                             train / test background.
    val_N, val_Z           : Neutron / proton numbers for validation nuclei.
                             Required when *val_metrics* is given.

    Returns
    -------
    fig, axes  (axes is a 1-D ndarray)
    """
    if shells is None:
        shells = SHELLS_MAGIC

    ncols = 2 if (residuals and true_vs_pred) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    axes = axes[0]  # flatten to 1-D

    tr_y_true = tr_metrics["y_true"]
    tr_y_pred = tr_metrics["y_pred"]
    te_y_true = te_metrics["y_true"]
    te_y_pred = te_metrics["y_pred"]

    lat_target = _fmt(target_name)

    # ── Panel 0: true vs predicted ─────────────────────────────────────────────
    if true_vs_pred:
        ax0 = axes[0]
        tr_rmse = tr_metrics["rmse"]
        te_rmse = te_metrics["rmse"]
        ax0.scatter(tr_y_true, tr_y_pred, s=8, alpha=0.4, color="steelblue",
                    label=f"Train  RMSE={tr_rmse:.2f}", marker="o")
        ax0.scatter(te_y_true, te_y_pred, s=8, alpha=0.6, color="tomato",
                    label=f"Test   RMSE={te_rmse:.2f}", marker="s")
        # Diagonal spans all data including val so no points float above the line
        all_true = [tr_y_true, te_y_true]
        if val_metrics is not None:
            val_rmse = val_metrics["rmse"]
            ax0.scatter(val_metrics["y_true"], val_metrics["y_pred"],
                        s=14, alpha=0.8, color="gold",
                        label=f"Val    RMSE={val_rmse:.2f}", zorder=5, marker="d")
            all_true.append(val_metrics["y_true"])
        lo = min(a.min() for a in all_true)
        hi = max(a.max() for a in all_true)
        ax0.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect fit")
        ax0.set_xlabel(f"True  {lat_target}")
        ax0.set_ylabel(f"Predicted  {lat_target}")
        ax0.set_title("True vs Predicted")
        ax0.legend(markerscale=2, fontsize=_FS_LEGEND)

    if residuals:
        # ── Panel 1: residual heatmap on the nuclear chart ─────────────────────
        # Train + test nuclei plotted as background (no outline).
        # Validation nuclei overlaid with black outline to separate them.
        ax1 = axes[1 if true_vs_pred else 0]

        tr_res = tr_y_pred - tr_y_true
        te_res = te_y_pred - te_y_true

        # Symmetric colour scale centred on zero, covering all splits.
        # Clipped to the 98th percentile of |residual| so a single outlier
        # does not compress the rest of the colourscale into a narrow band.
        all_res = np.concatenate([tr_res, te_res])
        if val_metrics is not None:
            val_res = val_metrics["y_pred"] - val_metrics["y_true"]
            all_res = np.concatenate([all_res, val_res])
        vmax = float(np.percentile(np.abs(all_res), 98))
        cnorm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        # Background: train + test
        sc = ax1.scatter(
            np.concatenate([tr_N, te_N]),
            np.concatenate([tr_Z, te_Z]),
            c=np.concatenate([tr_res, te_res]),
            cmap="RdBu_r", norm=cnorm,
            s=10, linewidths=0, alpha=0.9,
        )

        # Validation nuclei: same colour encoding, black outline
        if val_metrics is not None and val_N is not None and val_Z is not None:
            ax1.scatter(
                val_N, val_Z,
                c=val_res,
                cmap="RdBu_r", norm=cnorm,
                s=18, linewidths=0.6, edgecolors="black",
                zorder=5,
            )
            val_handle = mlines.Line2D(
                [], [], marker="o", linewidth=0,
                markerfacecolor="none", markeredgecolor="black",
                markeredgewidth=0.8, markersize=5,
                label="Val (AME2020)",
            )
            ax1.legend(handles=[val_handle], markerscale=1.5,
                       loc="lower right", fontsize=_FS_LEGEND)

        for m in shells:
            ax1.axvline(m, color=_SHELL_LINE_COLOR, lw=_SHELL_LINE_LW, ls=_SHELL_LINE_LS)
            ax1.axhline(m, color=_SHELL_LINE_COLOR, lw=_SHELL_LINE_LW, ls=_SHELL_LINE_LS)

        max_N = int(np.concatenate([tr_N, te_N,
                                    val_N if val_N is not None else tr_N]).max())
        max_Z = int(np.concatenate([tr_Z, te_Z,
                                    val_Z if val_Z is not None else tr_Z]).max())
        ax1.set_xlim(0, max_N)
        ax1.set_ylim(0, max_Z)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("Neutron number $N$")
        ax1.set_ylabel("Proton number $Z$")
        ax1.set_title(f"Residuals (MeV)")
        _inset_colorbar(fig, ax1, sc)

        # ── Old scatter approach (kept for reference, disabled) ────────────────
        # tr_res = tr_y_pred - tr_y_true
        # te_res = te_y_pred - te_y_true
        # for ax, xtr, xte, col in [
        #     (axes[1], tr_N, te_N, "N"),
        #     (axes[2], tr_Z, te_Z, "Z"),
        # ]:
        #     ax.scatter(xtr, tr_res, s=8, alpha=0.4, color="steelblue", label="Train")
        #     ax.scatter(xte, te_res, s=8, alpha=0.6, color="tomato",    label="Test")
        #     ax.axhline(0, color="k", lw=1, ls="--")
        #     for m in shells:
        #         ax.axvline(m, color=_SHELL_LINE_COLOR, lw=_SHELL_LINE_LW, ls=_SHELL_LINE_LS)
        #     ax.set_xlabel(_fmt(col))
        #     ax.set_ylabel("Residual (MeV)")
        #     ax.set_title(f"Residuals vs {_fmt(col)}")
        #     ax.legend(markerscale=2)

    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axes


# ── SHAP feature importance ────────────────────────────────────────────────────

def plot_shap(
    model,
    X_tr,
    X_te,
    feature_names: list[str],
    pred_fn=None,
    savefig: str | None = None,
    dpi: int = 500,
) -> None:
    """
    SHAP feature-importance summary for a PyTorch model.

    Produces two figures that mimic the native SHAP style:
      1. Bar chart of mean |SHAP value| per feature (global importance).
      2. Beeswarm summary plot (feature value coloured, one dot per test sample).

    Parameters
    ----------
    model         : Trained PyTorch nn.Module (eval mode set internally).
    X_tr          : Training tensor used as the SHAP background distribution.
    X_te          : Test tensor to explain.
    feature_names : List of feature name strings (length == X_te.shape[1]).
    pred_fn       : Optional callable ``model_output -> tensor`` that extracts
                    a single prediction tensor from the raw model output.
                    Required for models whose forward() returns a tuple, e.g.
                    MDN returns ``(pi, mu, sigma)`` which SHAP cannot handle.

                    MDN example::

                        pred_fn = lambda out: mdn_predict(*out)
    """
    # Lazy imports — torch/shap not required at module load time.
    import shap          # noqa: PLC0415
    import torch.nn as _nn

    # Wrap the model if it returns a tuple so GradientExplainer sees one tensor.
    if pred_fn is not None:
        class _PredWrapper(_nn.Module):
            def __init__(self, m, f):
                super().__init__()
                self.m, self.f = m, f
            def forward(self, x):
                return self.f(self.m(x))
        shap_model = _PredWrapper(model, pred_fn)
    else:
        shap_model = model

    shap_model.eval()
    explainer   = shap.GradientExplainer(shap_model, X_tr)
    shap_values = explainer.shap_values(X_te)        # list[ndarray] or ndarray

    # Normalise to 2-D (n_samples, n_features) regardless of SHAP version /
    # number of model outputs.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:          # (n_samples, n_features, n_outputs)
        shap_values = shap_values[:, :, 0]

    X_te_np     = X_te.numpy() if hasattr(X_te, "numpy") else np.asarray(X_te)
    latex_names = [_fmt(f) for f in feature_names]

    # Derive per-figure filenames from the savefig stem when saving.
    # e.g. "figures/shap.pdf" → "figures/shap_bar.pdf" + "figures/shap_beeswarm.pdf"
    if savefig is not None:
        from pathlib import Path as _Path
        _p    = _Path(savefig)
        _save = lambda tag: str(_p.with_stem(_p.stem + tag))  # noqa: E731
    else:
        _save = lambda tag: None  # noqa: E731

    # ── Figure 1: bar chart (mean |SHAP|) ─────────────────────────────────────
    shap.summary_plot(
        shap_values, X_te_np,
        feature_names=latex_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    if _save("_bar") is not None:
        plt.gcf().savefig(_save("_bar"), dpi=dpi, bbox_inches="tight")
    plt.show()

    # ── Figure 2: beeswarm (colour = feature value) ────────────────────────────
    shap.summary_plot(
        shap_values, X_te_np,
        feature_names=latex_names,
        show=False,
    )
    plt.tight_layout()
    if _save("_beeswarm") is not None:
        plt.gcf().savefig(_save("_beeswarm"), dpi=dpi, bbox_inches="tight")
    plt.show()


# ── Feature correlation analysis ───────────────────────────────────────────────

def plot_correlations(
    X: np.ndarray,
    feature_names: list[str],
    method: str = "spearman",
    scatter: bool = False,
    figsize: tuple[float, float] | None = None,
    savefig: str | None = None,
    dpi: int = 500,
) -> None:
    """
    Visualise pairwise feature correlations.

    Two complementary views:

    1. **Correlation heatmap** (always shown) — full symmetric matrix with colour
       encoding the chosen statistic, each cell annotated with its value.
       This immediately reveals multicollinearity (e.g. *A* and *Z* being nearly
       perfectly correlated makes one of them redundant).

    2. **Scatter matrix** (opt-in, ``scatter=True``) — pairwise scatter plots on
       the lower triangle, histograms on the diagonal, and the statistic on a
       coloured background in the upper triangle.  Captures non-linear structure
       and outliers that a single scalar misses.

    Why not a corner plot?  Corner plots (MCMC-style) include marginal KDE panels
    which add little information here and make the figure very tall for 8+
    features.  The scatter matrix gives the same pairwise view without the
    marginal overhead.

    Parameters
    ----------
    X            : Feature matrix of shape ``(n_samples, n_features)``.
    feature_names: Column labels, length must equal ``X.shape[1]``.
    method       : Correlation / dependence estimator.  Three choices:

                   * ``"spearman"`` (default) — rank-based monotonic correlation
                     *r* ∈ [–1, 1].  Robust to outliers and non-normal
                     distributions; appropriate for the discrete-valued Casimir
                     invariants used here.  Requires ``scipy``.

                   * ``"pearson"`` — standard linear *r* ∈ [–1, 1].  Assumes
                     normality and measures only linear association.

                   * ``"mic"`` — Maximal Information Coefficient ∈ [0, 1] from
                     information field theory (Reshef et al., Science 2011).
                     MIC detects *any* functional relationship (linear, periodic,
                     non-monotone) without assuming a functional form.  It equals
                     1 for a noiseless relationship and 0 for statistical
                     independence.  Note: MIC has *no sign* — it measures
                     strength of dependence only, not direction.  Requires
                     ``minepy`` (install with
                     ``pip install minepy --no-build-isolation``).

    scatter      : If True, also produce the full scatter matrix (default False).
    figsize      : Override the heatmap figure size.  The scatter matrix is always
                   auto-sized from the number of features.
    """
    n = X.shape[1]
    latex_names = [_fmt(f) for f in feature_names]

    # ── Compute correlation / dependence matrix ────────────────────────────────
    if method == "spearman":
        try:
            from scipy.stats import spearmanr
        except ImportError as exc:
            raise ImportError(
                "scipy is required for Spearman correlations: pip install scipy"
            ) from exc
        result = spearmanr(X)
        # Handle both old (tuple) and new (object) scipy return styles
        corr = np.array(
            result.statistic if hasattr(result, "statistic") else result[0]
        )
        if corr.ndim == 0:          # single-feature degenerate case
            corr = np.array([[1.0]])

    elif method == "mic":
        try:
            from minepy import pstats
        except ImportError as exc:
            raise ImportError(
                "minepy is required for MIC: "
                "pip install minepy --no-build-isolation"
            ) from exc
        try:
            from scipy.spatial.distance import squareform
        except ImportError as exc:
            raise ImportError(
                "scipy is required alongside minepy: pip install scipy"
            ) from exc
        # pstats expects (n_vars, n_samples); returns condensed upper-triangle
        mic_condensed, _ = pstats(X.T, alpha=0.6, c=15, est="mic_approx")
        corr = squareform(mic_condensed)
        np.fill_diagonal(corr, 1.0)   # MIC(x, x) = 1 by definition

    else:   # "pearson"
        corr = np.corrcoef(X.T)

    # ── Choose colormap and scale based on whether the metric is signed ────────
    is_mic  = (method == "mic")
    cmap_hm = "YlOrRd"    if is_mic else "RdBu_r"
    vmin_hm = 0.0          if is_mic else -1.0
    vmax_hm = 1.0
    cbar_lbl = "MIC"       if is_mic else f"{method.capitalize()} $r$"
    # threshold for white/dark text in heatmap cells
    _txt_thresh = 0.60     if is_mic else 0.65
    # map value → [0,1] for background colour in scatter-matrix upper triangle
    def _to_01(v):
        return v if is_mic else (0.5 + 0.5 * v)

    # ── Figure 1: annotated heatmap ────────────────────────────────────────────
    if not scatter:
        cell_in = max(0.65, 5.0 / n)                           # inches per cell
        hm_w    = n * cell_in + 2.0                            # extra room for colorbar
        hm_h    = n * cell_in + 0.8
        fig, ax = plt.subplots(figsize=figsize or (hm_w, hm_h))

        im = ax.imshow(corr, cmap=cmap_hm, vmin=vmin_hm, vmax=vmax_hm,
                       aspect="auto", interpolation="nearest")

        # Annotate every cell; flip to white text on strongly saturated colours
        for i in range(n):
            for j in range(n):
                v_ij = corr[i, j]
                txt_color = "white" if _to_01(v_ij) > _txt_thresh else "black"
                ax.text(j, i, f"{v_ij:.2f}",
                        ha="center", va="center",
                        fontsize=_FS_ANN, color=txt_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_lbl, fontsize=_FS_LABEL)
        cbar.ax.tick_params(labelsize=_FS_TICK)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(latex_names, rotation=45, ha="right", fontsize=_FS_TICK)
        ax.set_yticklabels(latex_names, fontsize=_FS_TICK)
        method_label = "MIC (Maximal Information Coefficient)" if is_mic else method.capitalize()
        # ax.set_title(f"Feature correlation matrix  ({method_label})", pad=10, fontsize=_FS_TITLE)
        fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.show()

    # ── Figure 2: scatter matrix ───────────────────────────────────────────────
    #   lower triangle : scatter plots
    #   diagonal       : histogram + feature label
    #   upper triangle : statistic on coloured background
    else:
        panel_in = max(1.4, 10.0 / n)
        fig2, axes = plt.subplots(n, n, figsize=figsize or (n * panel_in, n * panel_in))
        fig2.subplots_adjust(hspace=0.04, wspace=0.04)

        _cmap_scatter = plt.get_cmap(cmap_hm)

        for i in range(n):
            for j in range(n):
                ax2 = axes[i][j]
                # Remove all ticks/spines by default; re-enable on outer edges below
                ax2.tick_params(labelbottom=False, labelleft=False,
                                bottom=False, left=False, top=False, right=False)
                for spine in ax2.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("#cccccc")

                if i == j:
                    # ── diagonal: histogram ────────────────────────────────────
                    ax2.hist(X[:, i], bins=20, color="steelblue", alpha=0.75,
                             density=True, linewidth=0)
                    ax2.set_facecolor("#f4f6f8")

                elif i > j:
                    # ── lower triangle: scatter ────────────────────────────────
                    ax2.scatter(X[:, j], X[:, i],
                                s=4, alpha=0.35, color="steelblue", linewidths=0)
                    ax2.set_facecolor("#fafafa")

                else:
                    # ── upper triangle: statistic on coloured background ───────
                    v_ij  = corr[i, j]
                    bg    = _cmap_scatter(_to_01(v_ij))
                    ax2.set_facecolor(bg)
                    txt_c = "white" if _to_01(v_ij) > 0.55 else "black"
                    ax2.text(0.5, 0.5, f"{v_ij:.2f}",
                             transform=ax2.transAxes,
                             ha="center", va="center",
                             fontsize=_FS_ANN,
                             fontweight="bold", color=txt_c)

        # ── Outer-edge labels: top row → column names; left col → row names ──────
        for j in range(n):
            axes[0][j].set_title(latex_names[j], fontsize=_FS_LABEL,
                                 fontweight="bold", pad=4)
        for i in range(n):
            axes[i][0].set_ylabel(latex_names[i], fontsize=_FS_LABEL,
                                  fontweight="bold", labelpad=4)

        upper_label = "MIC" if is_mic else f"{method.capitalize()} $r$"
        #fig2.suptitle(f"Scatter matrix  |  {upper_label} in upper triangle", y=1.002, fontsize=_FS_TITLE)
        fig2.tight_layout()
        if savefig is not None:
            fig2.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.show()
