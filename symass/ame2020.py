"""
Parser for the AME2020 mass table  (mass_1.mas20).

Fortran format string (from file header):
  a1, i3, i5, i5, i5, 1x, a3, a4, 1x,
  f14.6, f12.6, f13.5, 1x, f10.5, 1x,
  a2, f13.5, f11.5,
  1x, i3, 1x, f13.6, f12.6

Column layout (0-indexed Python slices):
  ┌────────────┬──────────────────────────────────────────────────────────┐
  │  raw[...]  │  Field                                                   │
  ├────────────┼──────────────────────────────────────────────────────────┤
  │  [0]       │  Page marker (blank or '0')                    a1        │
  │  [1:4]     │  N – Z                                         i3        │
  │  [4:9]     │  N  (neutron number)                           i5        │
  │  [9:14]    │  Z  (proton number)                            i5        │
  │  [14:19]   │  A  (mass number)                              i5        │
  │  [19]      │  space                                         1x        │
  │  [20:23]   │  Element symbol                                a3        │
  │  [23:27]   │  Origin flag                                   a4        │
  │  [27]      │  space                                         1x        │
  │  [28:42]   │  Mass excess Δ [keV]  ('#' = extrapolated)     f14.6     │
  │  [42:54]   │  δΔ [keV]                                      f12.6     │
  │  [54:67]   │  Binding energy per nucleon BE/A [keV]         f13.5     │
  │  [67]      │  space                                         1x        │
  │  [68:78]   │  δ(BE/A) [keV]                                 f10.5     │
  │  [78]      │  space                                         1x        │
  │  [79:81]   │  β-decay sign                                  a2        │
  │  [81:94]   │  β-decay energy [keV]                          f13.5     │
  │  [94:105]  │  δ(β energy) [keV]                             f11.5     │
  │  [105]     │  space                                         1x        │
  │  [106:109] │  Atomic mass integer part [u]                  i3        │
  │  [109]     │  space                                         1x        │
  │  [110:123] │  Atomic mass fractional part [μu]              f13.6     │
  │  [123:135] │  δ(atomic mass) [μu]                           f12.6     │
  └────────────┴──────────────────────────────────────────────────────────┘

Atomic mass [u] = integer_part + fractional_part_μu / 1_000_000
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _safe_float(s: str) -> float:
    """
    Convert a fixed-width AME string to float.
    - '#' marks extrapolated values   → stripped, value returned.
    - '*' marks large uncertainties   → NaN.
    - Blank / empty                   → NaN.
    """
    s = s.strip()
    if not s or set(s) <= {"*", " "}:
        return np.nan
    s = s.replace("#", "").replace("*", "").replace("~", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _is_extrapolated(s: str) -> bool:
    return "#" in s


def _is_data_line(raw: str) -> bool:
    """True for lines that carry nuclear data (not headers or blank)."""
    if len(raw) < 100:
        return False
    if raw[0] not in (" ", "0"):
        return False
    # Reject header rows: N-Z field must be purely numeric/space/sign
    nz = raw[1:4]
    return bool(re.match(r"^[\s\-\d]+$", nz)) and nz.strip() != ""

# ── Reference values ───────────────────────────────────────────────────────────
# (Z, A) → (mass_excess_keV, BE_per_A_keV)
# Sources: AME2020 evaluation; BE/A for H-1 = 0 by convention.
_REFERENCE: dict[tuple[int, int], tuple[float, float]] = {
    (1,   1):  (  7_288.971,       0.000),   # ¹H
    (1,   2):  ( 13_135.722,   1_112.283),   # ²H  (deuteron)
    (2,   4):  (  2_424.916,   7_073.915),   # ⁴He (alpha)
    (6,  12):  (      0.000,   7_680.144),   # ¹²C  (mass standard)
    (8,  16):  ( -4_737.001,   7_976.206),   # ¹⁶O
    (20, 40):  (-34_846.000,   8_551.303),   # ⁴⁰Ca
    (26, 56):  (-60_605.000,   8_790.353),   # ⁵⁶Fe (most bound/nucleon)
    (82,208):  (-21_749.000,   7_867.447),   # ²⁰⁸Pb (doubly magic)
}

def _label(Z: int, A: int) -> str:
    return f"Z={Z},A={A}"

def verify(df: pd.DataFrame, tol_keV: float = 10.0) -> bool:
    """
    Compare parsed values against reference masses.

    Parameters
    ----------
    df      : Merged nuclear DataFrame (output of :func:`processing.merge.build`).
    tol_keV : Acceptable deviation [keV] from reference values.

    Returns
    -------
    bool
        ``True`` if all checks pass within *tol_keV*.

    Prints a formatted table; any failure prints a WARNING line.
    """
    print("\n  ─────────────────────────── Verify AME2020 ─────────────────────────────")
    print(f"  {'Nuclide':<8}  {'Δ parsed':>11}  {'Δ expected':>11}  "
          f"{'BE/A parsed':>12}  {'BE/A expect':>12}  {'Status'}")
    print("  " + "─" * 72)

    all_ok = True
    for (Z, A), (me_exp, be_exp) in sorted(_REFERENCE.items()):
        row = df[(df["Z"] == Z) & (df["A"] == A)]
        if row.empty:
            print(f"  {_label(Z,A):<8}  {'MISSING':>11}")
            all_ok = False
            continue

        me  = row["mass_excess_keV"].iloc[0]
        be  = row["BE_per_A_keV"].iloc[0]
        el  = row["element"].iloc[0]

        me_ok = not np.isnan(me) and abs(me - me_exp) < tol_keV
        be_ok = (be_exp == 0.0) or (not np.isnan(be) and abs(be - be_exp) < tol_keV)
        ok    = me_ok and be_ok
        status = "OK  " if ok else "FAIL"
        if not ok:
            all_ok = False

        me_str  = f"{me:+.1f}" if not np.isnan(me) else "NaN"
        be_str  = f"{be:.3f}"  if not np.isnan(be) else "NaN"
        print(f"  {el}-{A:<5}  {me_str:>11}  {me_exp:>+11.1f}  "
              f"{be_str:>12}  {be_exp:>12.3f}  {status}")

    print()
    if all_ok:
        print("  Verify successfully - all reference values match within tolerance.\n")
    else:
        print("  WARNING: failures detected — check column slice indices in parsers/ame2020.py.\n")
    return all_ok


# ── Generic parser (shared by AME2016 and AME2020) ─────────────────────────────

def _parse_ame_text(text: str, label: str = "AME") -> pd.DataFrame:
    """
    Parse a fixed-width AME mass table (works for AME2016 and AME2020 —
    both use the same Fortran column layout).

    Parameters
    ----------
    text  : Raw file text (latin-1 encoded string).
    label : Short tag used in progress messages, e.g. ``"AME2016"``.

    Returns
    -------
    pd.DataFrame  — one row per nuclide, sorted by (A, Z).
    """
    records: list[dict] = []

    for raw in text.splitlines():
        if not _is_data_line(raw):
            continue

        try:
            NZ = int(raw[1:4])
            N  = int(raw[4:9])
            Z  = int(raw[9:14])
            A  = int(raw[14:19])
            El = raw[20:23].strip()
        except ValueError:
            continue

        # ── Mass excess ────────────────────────────────────────────────────────
        me_raw          = raw[28:42]
        extrapolated    = _is_extrapolated(me_raw)
        mass_excess     = _safe_float(me_raw)
        mass_excess_unc = _safe_float(raw[42:54])

        # ── Binding energy per nucleon ─────────────────────────────────────────
        BE_A     = _safe_float(raw[54:67])
        BE_A_unc = _safe_float(raw[68:78])
        BE_total     = BE_A * A if not np.isnan(BE_A) else np.nan
        BE_total_unc = BE_A_unc * A if not np.isnan(BE_A_unc) else np.nan

        # ── β-decay energy ─────────────────────────────────────────────────────
        beta_sign       = raw[79:81].strip()
        beta_energy     = _safe_float(raw[81:94])
        beta_energy_unc = _safe_float(raw[94:105])

        # ── Atomic mass: integer [u] + fractional [μu] → u ────────────────────
        try:
            am_int = int(raw[106:109])
        except (ValueError, IndexError):
            am_int = None
        am_frac_val = _safe_float(raw[110:123]) if len(raw) > 123 else np.nan
        if am_int is not None and not np.isnan(am_frac_val):
            atomic_mass_u = am_int + am_frac_val / 1e6
        else:
            atomic_mass_u = np.nan
        am_unc_val        = _safe_float(raw[123:135]) if len(raw) > 135 else np.nan
        atomic_mass_unc_u = am_unc_val / 1e6 if not np.isnan(am_unc_val) else np.nan

        records.append({
            "Z":                     Z,
            "N":                     N,
            "A":                     A,
            "NZ":                    NZ,
            "element":               El,
            "extrapolated":          extrapolated,
            "mass_excess_keV":       mass_excess,
            "mass_excess_unc_keV":   mass_excess_unc,
            "BE_per_A_keV":          BE_A,
            "BE_per_A_unc_keV":      BE_A_unc,
            "BE_total_keV":          BE_total,
            "BE_total_unc_keV":      BE_total_unc,
            "beta_decay_energy_keV": beta_energy,
            "beta_energy_unc_keV":   beta_energy_unc,
            "beta_sign":             beta_sign,
            "atomic_mass_u":         atomic_mass_u,
            "atomic_mass_unc_u":     atomic_mass_unc_u,
        })

    df = pd.DataFrame(records).sort_values(["A", "Z"]).reset_index(drop=True)
    print(f"  {label:<10} {len(df):>5d} nuclides parsed")

    # ── Unit conversions ───────────────────────────────────────────────────────
    df["BE_per_A_MeV"]    = df["BE_per_A_keV"]   / 1e3
    df["BE_total_MeV"]    = df["BE_total_keV"]   / 1e3
    df["mass_excess_MeV"] = df["mass_excess_keV"] / 1e3

    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def parse(text: str) -> pd.DataFrame:
    """
    Parse the full text of ``mass_1.mas20`` (AME2020) into a DataFrame.

    Parameters
    ----------
    text : Raw file text (latin-1 encoded string).

    Returns
    -------
    pd.DataFrame
        One row per nuclide, sorted by (A, Z).  Columns:

        ``Z, N, A, NZ, element, extrapolated,
        mass_excess_keV, mass_excess_unc_keV,
        BE_per_A_keV, BE_per_A_unc_keV,
        BE_total_keV, BE_total_unc_keV,
        BE_per_A_MeV, BE_total_MeV, mass_excess_MeV,
        beta_decay_energy_keV, beta_energy_unc_keV, beta_sign,
        atomic_mass_u, atomic_mass_unc_u``
    """
    df = _parse_ame_text(text, label="AME2020:")
    print(f"\n  All columns:")
    for col in df.columns:
        print(f"    {col}")
    verify(df)
    return df


def parse_ame2016(text: str) -> pd.DataFrame:
    """
    Parse the full text of ``mass16.txt`` (AME2016) into a DataFrame.

    Same column layout as AME2020; only the nuclear data differ.
    The verify() check is intentionally skipped — reference values are
    from AME2020 and may differ slightly from AME2016.

    Parameters
    ----------
    text : Raw file text (latin-1 encoded string).

    Returns
    -------
    pd.DataFrame  — same schema as :func:`parse`.
    """
    return _parse_ame_text(text, label="AME2016:")
