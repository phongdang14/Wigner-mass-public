"""
Parser for NUBASE2020  (nubase_4.mas20).

Key columns extracted (1-indexed, inclusive, from NUBASE2020 documentation):
  ┌───────────┬──────────────────────────────────────────────────────────┐
  │  Col(s)   │  Field                                                   │
  ├───────────┼──────────────────────────────────────────────────────────┤
  │   1 –  3  │  A  (mass number)                                        │
  │   5 –  7  │  Z  (atomic number)                                      │
  │   8       │  i  isomer flag: 0=gs, 1,2=isomers, 3–6=levels,          │
  │           │     5=resonance, 8,9=IAS                                 │
  │  70 – 78  │  Half-life value  (# from systematics)                   │
  │  79 – 80  │  Half-life unit                                          │
  │  82 – 88  │  Half-life uncertainty string                            │
  │  89 –102  │  Spin-parity Jπ  (* measured; # systematics)             │
  │ 120 –209  │  Decay modes and branching ratios                        │
  └───────────┴──────────────────────────────────────────────────────────┘

Only ground-state rows (isomer flag = '0' or blank) are returned.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ── Half-life unit → seconds ───────────────────────────────────────────────────
_HL_TO_S: dict[str, float] = {
    "ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15,
    "ps": 1e-12, "ns": 1e-9,  "us": 1e-6,  "ms": 1e-3,
    "s":  1.0,
    "m":  60.0,
    "h":  3_600.0,
    "d":  86_400.0,
    "y":  3.155_76e7,
    "ky": 3.155_76e10,
    "My": 3.155_76e13,
    "Gy": 3.155_76e16,
    "Ty": 3.155_76e19,
    "Py": 3.155_76e22,
    "Ey": 3.155_76e25,
    "Zy": 3.155_76e28,
    "Yy": 3.155_76e31,
}


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _halflife_to_s(val_str: str, unit_str: str) -> float:
    """Convert NUBASE half-life value + unit to seconds."""
    val_raw  = val_str.strip()
    unit_str = unit_str.strip()
    # "stbl" and "p-unst"/"n-unst" appear in the T field, not the unit field
    if val_raw == "stbl":
        return np.inf
    if val_raw in ("p-unst", "n-unst"):
        return np.nan
    val_str = val_raw.lstrip("><~").replace("#", "").replace("*", "")
    if not val_str or not unit_str:
        return np.nan
    try:
        val = float(val_str)
    except ValueError:
        return np.nan
    factor = _HL_TO_S.get(unit_str, np.nan)
    return val * factor if not np.isnan(factor) else np.nan


def _partial_halflife(total: float, branch_pct: float) -> float:
    """
    Partial half-life from total half-life and branching ratio.
    T½_partial = T½_total × (100 / BR%)
    """
    if np.isnan(branch_pct) or branch_pct <= 0:
        return np.nan
    if np.isnan(total) or np.isinf(total):
        return np.nan
    return total * 100.0 / branch_pct


def _parse_decay_modes(s: str) -> dict[str, float]:
    """
    Extract branching ratios (%) from a NUBASE decay-mode string.

    Handles:
    - ``A=5.3``         → alpha 5.3 %
    - ``B-=100``        → β⁻  100 %
    - ``B+=60;EC=40``   → β⁺ 60 %, EC 40 %
    - ``A``             → alpha 100 % (dominant, no percentage given)
    - ``IS``            → stable (all NaN)

    Returns
    -------
    dict with keys: alpha_pct, beta_minus_pct, beta_plus_pct, EC_pct, SF_pct
    """
    result: dict[str, float] = {
        "alpha_pct":     np.nan,
        "beta_minus_pct": np.nan,
        "beta_plus_pct": np.nan,
        "EC_pct":        np.nan,
        "SF_pct":        np.nan,
    }
    if not s:
        return result

    _PATTERNS = [
        (r"A",    "alpha_pct"),
        (r"B-",   "beta_minus_pct"),
        (r"B\+",  "beta_plus_pct"),
        (r"EC",   "EC_pct"),
        (r"SF",   "SF_pct"),
    ]

    for tok in re.split(r"[;,\s]+", s.strip()):
        tok = tok.strip()
        for pattern, key in _PATTERNS:
            # With explicit percentage: "A=5.3"
            m = re.match(rf"^{pattern}=([\d.]+)$", tok, re.IGNORECASE)
            if m:
                result[key] = float(m.group(1))
                break
            # Dominant mode (no percentage): "A" alone
            if re.fullmatch(pattern, tok, re.IGNORECASE):
                result[key] = 100.0
                break

    return result


# ── Reference half-lives ───────────────────────────────────────────────────────
# (Z, A) → (half_life_s, label)
# Expected values derived from NUBASE2020 / CODATA; np.inf = stable.
_REFERENCE_HL: dict[tuple[int, int], tuple[float, str]] = {
    (1,   3): (3.885e8,   "3H   tritium"),      # 12.32 y
    (6,  12): (np.inf,    "12C  stable"),
    (6,  14): (1.808e11,  "14C  carbon-14"),    # 5730 y
    (27, 60): (1.663e8,   "60Co"),              # 5.2713 y
    (55,137): (9.489e8,   "137Cs"),             # 30.08 y
    (82,208): (np.inf,    "208Pb stable"),
    (88,226): (5.049e10,  "226Ra Curie"),       # 1600 y
    (92,238): (1.410e17,  "238U  primordial"),  # 4.468 Gy
}

_TOL_REL = 0.01   # 1 % relative tolerance for unstable nuclides


def verify(df: pd.DataFrame) -> bool:
    """
    Spot-check parsed half-lives against known reference values.

    Uses a 1 % relative tolerance for unstable nuclides and an exact
    ``isinf`` check for stable ones.  Mirrors the structure of
    :func:`ame2020.verify`.

    Returns
    -------
    bool
        ``True`` if all checks pass.
    """
    print("\n  ──────────────────── Verify NUBASE2020 ──────────────────")
    print(f"  {'Nuclide':<14}  {'T½ parsed (s)':>14}  {'T½ expected (s)':>15}  {'Status'}")
    print("  " + "─" * 57)

    all_ok = True
    for (Z, A), (hl_exp, label) in sorted(_REFERENCE_HL.items()):
        row = df[(df["Z"] == Z) & (df["A"] == A)]
        if row.empty:
            print(f"  {label:<14}  {'MISSING':>14}")
            all_ok = False
            continue

        hl = row["half_life_s"].iloc[0]

        if np.isinf(hl_exp):
            ok = np.isinf(hl)
        elif np.isnan(hl):
            ok = False
        else:
            ok = abs(hl - hl_exp) / hl_exp < _TOL_REL

        status = "OK  " if ok else "FAIL"
        if not ok:
            all_ok = False

        hl_str     = "stable" if np.isinf(hl)     else (f"{hl:.3e}"     if not np.isnan(hl) else "NaN")
        hl_exp_str = "stable" if np.isinf(hl_exp) else f"{hl_exp:.3e}"
        print(f"  {label:<14}  {hl_str:>14}  {hl_exp_str:>15}  {status}")

    print()
    if all_ok:
        print("  Verify successfully - all reference half-lives match within tolerance.\n")
    else:
        print("  WARNING: failures detected — check column slice indices in nubase2020.py.\n")
    return all_ok


# ── Public API ─────────────────────────────────────────────────────────────────

def parse(text: str) -> pd.DataFrame:
    """
    Parse the full text of ``nubase_1.mas20`` into a
    :class:`~pandas.DataFrame`.  Only ground-state rows are returned.

    Parameters
    ----------
    text : Raw file text (latin-1 encoded string).

    Returns
    -------
    pd.DataFrame
        One row per ground-state nuclide, sorted by (A, Z).  Columns:

        ``Z, N, A,
        half_life_s, half_life_unc_str, spin_parity,
        alpha_branch_pct, beta_minus_branch_pct,
        beta_plus_branch_pct, EC_branch_pct, SF_branch_pct,
        alpha_partial_hl_s, beta_minus_partial_hl_s, beta_plus_partial_hl_s``
    """
    records: list[dict] = []

    for raw in text.splitlines():
        if len(raw) < 50:
            continue
        # Skip non-data lines
        if not raw[0].isdigit():
            continue

        try:
            A = int(raw[0:3])
            Z = int(raw[4:7])
        except ValueError:
            continue

        # Ground states only: i=0 (digit '0') or space; skip isomers/levels/IAS
        isomer = raw[7] if len(raw) > 7 else " "
        if isomer not in ("0", " "):
            continue

        N = A - Z

        # ── Half-life ──────────────────────────────────────────────────────────
        hl_val  = raw[69:78].strip() if len(raw) > 78 else ""
        hl_unit = raw[78:80].strip() if len(raw) > 80 else ""
        hl_unc  = raw[81:88].strip() if len(raw) > 88 else ""
        hl_s    = _halflife_to_s(hl_val, hl_unit)

        # ── Spin-parity ────────────────────────────────────────────────────────
        spin_parity = raw[88:102].strip() if len(raw) > 102 else ""

        # ── Decay modes ────────────────────────────────────────────────────────
        decay_str = raw[119:209].strip() if len(raw) > 119 else ""
        modes     = _parse_decay_modes(decay_str)

        records.append({
            "Z":                       Z,
            "N":                       N,
            "A":                       A,
            "half_life_s":             hl_s,
            "half_life_unc_str":       hl_unc,
            "spin_parity":             spin_parity,
            "alpha_branch_pct":        modes["alpha_pct"],
            "beta_minus_branch_pct":   modes["beta_minus_pct"],
            "beta_plus_branch_pct":    modes["beta_plus_pct"],
            "EC_branch_pct":           modes["EC_pct"],
            "SF_branch_pct":           modes["SF_pct"],
            "alpha_partial_hl_s":      _partial_halflife(hl_s, modes["alpha_pct"]),
            "beta_minus_partial_hl_s": _partial_halflife(hl_s, modes["beta_minus_pct"]),
            "beta_plus_partial_hl_s":  _partial_halflife(hl_s, modes["beta_plus_pct"]),
        })

    df = pd.DataFrame(records).sort_values(["A", "Z"]).reset_index(drop=True)
    print(f"  NUBASE2020:  {len(df):>5d} ground-state nuclides parsed")

    print(f"\n  All columns:")
    for col in df.columns:
        print(f"    {col}")

    verify(df)

    return df
