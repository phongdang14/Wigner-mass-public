"""
Tools for managing datasets: download/fetch, clipping, save and load data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .constants import (
    AME2020_URL, AME2016_URL, NUBASE2020_URL,
    AME2020_CACHE, AME2016_CACHE, NUBASE2020_CACHE,
    OUTPUT_CSV, OUTPUT_PKL,
)

# Default data directory — resolved relative to this file, not the caller's CWD
DATA_DIR = Path(__file__).parent.parent / "data"


# ── Download ───────────────────────────────────────────────────────────────────

def download_text(url: str, cache_path: Path, encoding: str = "latin-1",
                  force: bool = False) -> str:
    """
    Fetch *url* and return its text content.
    If *cache_path* already exists the download is skipped and the cached
    file is returned instead.  Pass ``force=True`` to re-download even when
    the cache exists.

    Parameters
    ----------
    url        : Remote URL to fetch.
    cache_path : Local path where the file is stored after download.
    encoding   : Text encoding (AME/NUBASE files use latin-1).
    force      : If True, ignore the cache and always re-download.

    Returns
    -------
    str
        Full file text.
    """
    if cache_path.exists() and not force:
        print(f"  [cached]  {cache_path.name}")
        return cache_path.read_text(encoding=encoding)

    print(f"  [fetch]   {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    text = resp.content.decode(encoding)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding=encoding)
    print(f"  [saved]   {cache_path}")
    return text


def fetch_ame2020(data_dir: Path = DATA_DIR, force: bool = False) -> str:
    """Download (or load from cache) the AME2020 mass table."""
    return download_text(AME2020_URL, data_dir / AME2020_CACHE, force=force)


def fetch_nubase2020(data_dir: Path = DATA_DIR, force: bool = False) -> str:
    """Download (or load from cache) the NUBASE2020 file."""
    return download_text(NUBASE2020_URL, data_dir / NUBASE2020_CACHE, force=force)


def fetch_ame2016(data_dir: Path = DATA_DIR, force: bool = False) -> str:
    """Download (or load from cache) the AME2016 mass table."""
    return download_text(AME2016_URL, data_dir / AME2016_CACHE, force=force)


# ── Save ────────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, data_dir: Path = DATA_DIR) -> None:
    """
    Save *df* to both CSV and pickle inside *data_dir*.

    The pickle is used for fast re-loading during analysis; the CSV is
    the human-readable audit trail.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / OUTPUT_CSV
    pkl_path = data_dir / OUTPUT_PKL
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"  CSV    → {csv_path}")
    print(f"  Pickle → {pkl_path}")


# ── Filter ──────────────────────────────────────────────────────────────

def clip(df: pd.DataFrame,
            Amin: int | None = None, Amax: int | None = None,
            Zmin: int | None = None, Zmax: int | None = None,
            Nmin: int | None = None, Nmax: int | None = None) -> pd.DataFrame:
    """
    Return a copy of *df* keeping only nuclei within the specified ranges.
    Any bound left as None is treated as unbounded.
    """
    mask = pd.Series(True, index=df.index)
    if Amin is not None: mask &= df["A"] >= Amin
    if Amax is not None: mask &= df["A"] <= Amax
    if Zmin is not None: mask &= df["Z"] >= Zmin
    if Zmax is not None: mask &= df["Z"] <= Zmax
    if Nmin is not None: mask &= df["N"] >= Nmin
    if Nmax is not None: mask &= df["N"] <= Nmax
    return df[mask].reset_index(drop=True)


# ── Normalizer ──────────────────────────────────────────────────────────

class Normalizer:
    """
    Z-score normaliser for features and targets.

    Fitted on training data; call transform_* on any split and inverse_y
    to recover predictions in the original unit.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X_mean = X.mean(axis=0)          # (n_features,)
        self.X_std  = X.std(axis=0)  + 1e-8   # guard against constant columns
        self.y_mean = y.mean(axis=0)           # (n_targets,)
        self.y_std  = y.std(axis=0)  + 1e-8

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


# ── Load ────────────────────────────────────────────────────────────────

def load_dataframe(
    data_dir: Path = DATA_DIR,
    features: list[str] | None = None,
    targets:  list[str] | None = None,
    keep_extrapolated: bool = False,
    Amin: int | None = None, Amax: int | None = None,
    Zmin: int | None = None, Zmax: int | None = None,
    Nmin: int | None = None, Nmax: int | None = None,
) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the merged nuclear DataFrame from the pickle produced by
    :func:`save_dataframe`.

    Parameters
    ----------
    data_dir : Path
        Directory that holds the pickle file.
    features : list[str] | None
        If provided, also return a float32 array ``X`` of shape
        ``(n_samples, n_features)`` with those columns.
    targets : list[str] | None
        If provided, also return a float32 array ``y`` of shape
        ``(n_samples, n_targets)`` with those columns.
    keep_extrapolated : bool
        If True, include nuclei whose masses are extrapolated (default False).
    Amin, Amax : int | None
        Inclusive bounds on mass number *A*.
    Zmin, Zmax : int | None
        Inclusive bounds on proton number *Z*.
    Nmin, Nmax : int | None
        Inclusive bounds on neutron number *N*.

    Returns
    -------
    pd.DataFrame
        When neither *features* nor *targets* is given.
    (X, y, df) : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        When at least one of *features* or *targets* is given.
        Missing argument defaults to an empty array with 0 columns.

    Raises
    ------
    FileNotFoundError
        If the pickle does not exist yet (run src/build.py first).
    KeyError
        If any requested feature or target column is absent from the DataFrame.
    """
    pkl_path = data_dir / OUTPUT_PKL
    csv_path = data_dir / OUTPUT_CSV

    if not pkl_path.exists() and not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {pkl_path}. Run src/build.py first."
        )

    # Try the fast pickle first; fall back to CSV when the pickle was written
    # with a different numpy/pandas version (common numpy 1.x ↔ 2.x mismatch).
    df = None
    if pkl_path.exists():
        try:
            df = pd.read_pickle(pkl_path)
        except Exception as exc:
            print(
                f"  [warn] Could not load pickle ({type(exc).__name__}: {exc}).\n"
                f"         Falling back to CSV — consider re-running src/build.py "
                f"to rebuild the pickle in this environment."
            )

    if df is None:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Neither pickle nor CSV found at {data_dir}. Run src/build.py first."
            )
        df = pd.read_csv(csv_path)
        # CSV stores booleans as strings — restore the correct dtype
        if "extrapolated" in df.columns:
            df["extrapolated"] = df["extrapolated"].map(
                {True: True, False: False, "True": True, "False": False}
            ).astype(bool)

    # Apply extrapolation filter before anything else
    target_cols = targets or []
    valid_mask  = df[target_cols].notna().all(axis=1) if target_cols else pd.Series(True, index=df.index)
    if not keep_extrapolated:
        valid_mask &= ~df["extrapolated"]
    df = df[valid_mask].reset_index(drop=True)

    # Apply optional nuclear-number clipping
    if any(v is not None for v in (Amin, Amax, Zmin, Zmax, Nmin, Nmax)):
        df = clip(df, Amin=Amin, Amax=Amax, Zmin=Zmin, Zmax=Zmax, Nmin=Nmin, Nmax=Nmax)

    if features is None and targets is None:
        return df

    missing = [c for c in (features or []) + (targets or []) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataset: {missing}")

    X = df[features].to_numpy(dtype=np.float32) if features else np.empty((len(df), 0), dtype=np.float32)
    y = df[targets].to_numpy(dtype=np.float32)  if targets  else np.empty((len(df), 0), dtype=np.float32)

    return X, y, df


def load_data(
    features: list[str],
    targets:  list[str],
    data_dir: Path = DATA_DIR,
    keep_extrapolated: bool = False,
    Amin: int | None = None, Amax: int | None = None,
    Zmin: int | None = None, Zmax: int | None = None,
    Nmin: int | None = None, Nmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: load the pre-built dataset and return arrays
    ready for model training.

    Parameters
    ----------
    features : list[str]
        Feature column names.
    targets : list[str]
        Target column names.
    data_dir : Path
        Directory that holds the pickle file.
    keep_extrapolated : bool
        If True, include extrapolated masses (default False).
    Amin, Amax : int | None
        Inclusive bounds on mass number *A*.
    Zmin, Zmax : int | None
        Inclusive bounds on proton number *Z*.
    Nmin, Nmax : int | None
        Inclusive bounds on neutron number *N*.

    Returns
    -------
    X          : np.ndarray  (n, n_features)  float32
    y          : np.ndarray  (n, n_targets)   float32
    N          : np.ndarray  (n,)             int32
    Z          : np.ndarray  (n,)             int32
    ame_source : np.ndarray  (n,)             str —  ``"AME2016"`` or
                 ``"AME2020"``.  If the dataset was built without the AME2016
                 reference (old build), all entries are ``"AME2016"``.
    """
    df = load_dataframe(
        data_dir=data_dir,
        keep_extrapolated=keep_extrapolated,
        Amin=Amin, Amax=Amax,
        Zmin=Zmin, Zmax=Zmax,
        Nmin=Nmin, Nmax=Nmax,
    )

    missing = [c for c in features + targets if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataset: {missing}")

    X = df[features].to_numpy(dtype=np.float32)
    y = df[targets].to_numpy(dtype=np.float32)
    N = df["N"].to_numpy(dtype=np.int32)
    Z = df["Z"].to_numpy(dtype=np.int32)

    if "ame_source" in df.columns:
        ame_source = df["ame_source"].to_numpy()
    else:
        import warnings
        warnings.warn(
            "'ame_source' column not found — dataset was built without AME2016 "
            "reference. Re-run src/build.py. Defaulting all entries to 'AME2016'.",
            stacklevel=2,
        )
        ame_source = np.full(len(df), "AME2016")

    return X, y, N, Z, ame_source
