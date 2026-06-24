"""
Add SU(4) columns: P1, P2, P3, C2_4, C3_4, C4_4
Add SU(3) columns: shell, lam, mu, C2_3, C3_3
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_su4_su3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment *df* with SU(4) and SU(3) columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new columns appended.
    """
    df = df.copy()

    # ── Isospin ────────────────────────────────────────────────────────────────
    df["T_z"] = (df["N"] - df["Z"]) / 2.0

    # ── SU(3) x SU(4) irrep labels ────────────────────────────────────────────
    df["shell"], df["Nhw"], df["lam"], df["mu"], df["P1"], df["P2"], df["P3"] = _irrep(df)

    # ── SU(4) Casimir invariants ───────────────────────────────────────────────
    df["C2_4"] = _C2_4(df)
    df["C3_4"] = _C3_4(df)
    df["C4_4"] = _C4_4(df)

    # ── SU(3) Casimir invariants ───────────────────────────────────────────────
    df["C2_3"] = _C2_3(df)
    df["C3_3"] = _C3_3(df)

    print("SU(3) and SU(4) columns added.")
    verify(df)
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _irr(A: int, N: int, Z: int) -> tuple[float, float, float, float, float, float, float]:
    """
    Find the leading U(4) irrep [n14, n24, n34, n44] for a nucleus (A, N, Z),
    then derive (Nhw, lam, mu, P1, P2, P3).

    The leading irrep is the valid U(4) irrep with the lowest SU(4) Casimir C2.
    Valid means: rows are non-increasing, sum to the valence count a, fit within
    the shell degeneracy, and isospin |N-Z| is compatible via accept().
    """

    # Enumerate valid U(4) irreps; keep the one with lowest C2
    TT      = abs(N - Z)            
    best    = None
    best_c2 = float("inf")

    for n14 in range(A, -1, -1):
        for n24 in range(n14, -1, -1):
            if n14 + n24 > A:
                continue
            for n34 in range(n24, -1, -1):
                if n14 + n24 + n34 > A:
                    continue
                n44 = A - (n14 + n24 + n34)
                if n44 <= n34 and n44 >= 0 and accept(n14, n24, n34, n44, TT):
                    p1 = float(n14 - n24)
                    p2 = float(n24 - n34)
                    p3 = float(n34 - n44)
                    c2 = (1.5*p1**2 + 2.0*p2**2 + 1.5*p3**2
                          + 2.0*p1*p2 + 2.0*p2*p3 + p1*p3
                          + 6.0*p1 + 8.0*p2 + 6.0*p3)
                    if c2 < best_c2:
                        best_c2 = c2
                        best = (n14, n24, n34, n44)

    if best is None:
        print(f"A:{A}, N:{N}, Z:{Z}")
    assert best is not None

    n14, n24, n34, n44 = best
    P1 = float(n14 - n24)
    P2 = float(n24 - n34)
    P3 = float(n34 - n44)

    cnt4 = n44
    cnt3 = n34 - n44
    cnt2 = n24 - n34
    cnt1 = n14 - n24

    total_levels = cnt4 + cnt3 + cnt2 + cnt1
    if total_levels == 0:
        return 0.0, 0.0, 0.0, 0.0, P1, P2, P3

    # Accumulate single-particle levels shell by shell (η=0,1,2,…),
    # descending lex within each shell, until we have total_levels slots.
    levels_all: list[tuple[int, int, int]] = []
    eta = 0
    while len(levels_all) < total_levels:
        shell_levels = sorted(
            [(n1, n2, eta - n1 - n2)
             for n1 in range(eta + 1)
             for n2 in range(eta - n1 + 1)],
            key=lambda x: (-x[0], -x[1])
        )
        levels_all.extend(shell_levels)
        eta += 1

    levels_all = levels_all[:total_levels]
    eta_out = eta - 1  # highest shell used

    occupations = [4]*cnt4 + [3]*cnt3 + [2]*cnt2 + [1]*cnt1
    n13 = n23 = n33 = 0
    for (sp_n1, sp_n2, sp_n3), occ in zip(levels_all, occupations):
        n13 += occ * sp_n1
        n23 += occ * sp_n2
        n33 += occ * sp_n3

    return float(eta_out), float(n13 + n23 + n33), float(n13 - n23), float(n23 - n33), P1, P2, P3


def accept(n14: int, n24: int, n34: int, n44: int, TT: int) -> bool:
    tt_max = n14 + n24 - n34 - n44
    tt_min = tt_max % 2
    return tt_min <= TT <= tt_max


def _irrep(df: pd.DataFrame) -> tuple[pd.Series, ...]:
    idx = df.index
    A = df["A"].to_numpy(dtype=int)
    N = df["N"].to_numpy(dtype=int)
    Z = df["Z"].to_numpy(dtype=int)

    shell = np.empty(len(df), dtype=float)
    Nhw   = np.empty(len(df), dtype=float)
    lam   = np.empty(len(df), dtype=float)
    mu    = np.empty(len(df), dtype=float)
    P1    = np.empty(len(df), dtype=float)
    P2    = np.empty(len(df), dtype=float)
    P3    = np.empty(len(df), dtype=float)

    for i in range(len(df)):
        shell[i], Nhw[i], lam[i], mu[i], P1[i], P2[i], P3[i] = _irr(int(A[i]), int(N[i]), int(Z[i]))

    return (pd.Series(shell, index=idx), pd.Series(Nhw, index=idx),
            pd.Series(lam, index=idx),   pd.Series(mu, index=idx),
            pd.Series(P1, index=idx),    pd.Series(P2, index=idx),
            pd.Series(P3, index=idx))


# ── SU(4) Casimirs ─────────────────────────────────────────────────────────────

def _C2_4(df: pd.DataFrame) -> pd.Series:
    P1, P2, P3 = df["P1"], df["P2"], df["P3"]
    return (1.5 * P1**2 + 2.0 * P2**2 + 1.5 * P3**2
            + 2.0 * P1 * P2 + 2.0 * P2 * P3 + P1 * P3
            + 6.0 * P1 + 8.0 * P2 + 6.0 * P3)


def _C3_4(df: pd.DataFrame) -> pd.Series:
    P1, P2, P3 = df["P1"], df["P2"], df["P3"]
    return 1.5 * (P1 - P3) * (
        P1**2 + P3**2
        + 2.0 * (P1 * P2 + P2 * P3 + P1 * P3)
        + 2.0 * (3.0 * P1 + 2.0 * P2 + 3.0 * P3 + 4.0)
    )


def _C4_4(df: pd.DataFrame) -> pd.Series:
    P1, P2, P3 = df["P1"], df["P2"], df["P3"]
    return (
        21.0 * P1**4 + 16.0 * P2**4 + 21.0 * P3**4
        + 4.0  * P1**3 * (14.0 * P2 + 7.0 * P3 + 42.0)
        + 4.0  * P3**3 * (14.0 * P2 + 7.0 * P1 + 42.0)
        + 32.0 * P2**3 * (4.0 + P1 + P3)
        + 72.0 * P2**2 * (P1**2 + P3**2)
        + 30.0 * P1**2 * P3**2
        + 24.0 * P1 * P2 * P3 * (3.0 * P1 + 2.0 * P2 + 3.0 * P3)
        + 288.0 * P2**2 * (P1 + P3)
        + 24.0 * P1**2 * (16.0 * P2 + 9.0 * P3)
        + 24.0 * P3**2 * (16.0 * P2 + 9.0 * P1)
        + 384.0 * P1 * P2 * P3
        + 576.0 * (P1**2 + P3**2)
        + 512.0 * P2**2
        + 896.0 * P2 * (P1 + P3)
        + 640.0 * P1 * P3
        + 960.0 * (P1 + P3)
        + 1024.0 * P2
    ) / 64.0


# ── SU(3) Casimirs ─────────────────────────────────────────────────────────────

def _C2_3(df: pd.DataFrame) -> pd.Series:
    lam, mu = df["lam"], df["mu"]
    return (lam * lam + lam * mu + mu * mu + 3.0 * lam + 3.0 * mu)


def _C3_3(df: pd.DataFrame) -> pd.Series:
    lam, mu = df["lam"], df["mu"]
    return (lam - mu) * (lam + 2.0 * mu + 3.0) * (2.0 * lam + mu + 3.0) / 9.0


# ── Verify ─────────────────────────────────────────────────────────────────────

# (Z, A) → (shell, lam, mu, label)
_REFERENCE_SU3: dict[tuple[int, int], tuple[int, int, int, str]] = {
    (2,   4): (0, 0, 0, "4He  closed s-shell"),
    (3,   6): (1, 2, 0, "6Li  p-shell 2 valence"),
    (4,   8): (1, 4, 0, "8Be  p-shell 4 valence"),
    (6,  12): (1, 0, 4, "12C  p-shell 8 valence"),
    (8,  16): (1, 0, 0, "16O  closed p-shell"),
    (10, 20): (2, 8, 0, "20Ne sd-shell 4 valence"),
    (12, 24): (2, 8, 4, "24Mg sd-shell 8 valence"),
    (16, 32): (2, 4, 8, "32S  sd-shell 16 valence"),
    (20, 40): (2, 0, 0, "40Ca closed sd-shell"),
}


def verify(df: pd.DataFrame) -> bool:
    """Spot-check (shell, lam, mu) against known SU(3) leading irreps."""
    print("\n  ──────────────────── Verify SU(3) ──────────────────────")
    print(f"  {'Nuclide':<22}  {'shell':>5}  {'lam':>4}  {'mu':>4}  "
          f"{'shell exp':>9}  {'lam exp':>7}  {'mu exp':>6}  {'Status'}")
    print("  " + "─" * 72)

    all_ok = True
    for (Z, A), (sh_exp, lam_exp, mu_exp, label) in sorted(_REFERENCE_SU3.items()):
        row = df[(df["Z"] == Z) & (df["A"] == A)]
        if row.empty:
            print(f"  {label:<22}  {'MISSING':>5}")
            all_ok = False
            continue

        sh  = int(row["shell"].iloc[0])
        lam = int(row["lam"].iloc[0])
        mu  = int(row["mu"].iloc[0])
        ok  = (sh == sh_exp) and (lam == lam_exp) and (mu == mu_exp)
        if not ok:
            all_ok = False

        status = "OK  " if ok else "FAIL"
        print(f"  {label:<22}  {sh:>5}  {lam:>4}  {mu:>4}  "
              f"{sh_exp:>9}  {lam_exp:>7}  {mu_exp:>6}  {status}")

    print()
    if all_ok:
        print("  Verify successfully — all SU(3) leading irreps match.\n")
    else:
        print("  WARNING: failures detected.\n")
    return all_ok

# ── A formula to determine the dominant irrep, however, it doesn't seem to be right ─────

def su4_irrep(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute the SU(4) leading irrep labels (P1, P2, P3) for each nuclide.
    Ref: P.V. Isacker, O. Juillet and B.K. Gjelsten, Foundations of Physics 27 (1997) 1047

    Rules based on (N mod 2, Z mod 2):
      even-even        : P1=0,  P2=|N-Z|/2,       P3=0
      even-odd         : P1=1,  P2=|N-Z|/2-1/2,     P3=0
      odd-even         : P1=0,  P2=|N-Z|/2-1/2,     P3=1
      odd-odd, N==Z    : P1=0,  P2=1,              P3=0
      odd-odd, N!=Z    : P1=1,  P2=|N-Z|/2-1,     P3=1
    """
    N = df["N"].to_numpy(dtype=int)
    Z = df["Z"].to_numpy(dtype=int)
    idx = df.index

    N_even = (N % 2 == 0)
    Z_even = (Z % 2 == 0)

    P1 = np.zeros(len(df), dtype=float)
    P2 = np.zeros(len(df), dtype=float)
    P3 = np.zeros(len(df), dtype=float)

    # even-even
    m = N_even & Z_even
    P1[m] = 0
    P2[m] = np.abs(N[m] - Z[m]) / 2
    P3[m] = 0

    # even-odd
    m = N_even & ~Z_even
    P1[m] = 1
    P2[m] = np.abs(N[m] - Z[m]) / 2 - 1 / 2
    P3[m] = 0

    # odd-even
    m = ~N_even & Z_even
    P1[m] = 0
    P2[m] = np.abs(N[m] - Z[m]) / 2 - 1 / 2
    P3[m] = 1

    # odd-odd, N == Z
    m = ~N_even & ~Z_even & (N == Z)
    P1[m] = 0
    P2[m] = 1
    P3[m] = 0

    # odd-odd, N != Z
    m = ~N_even & ~Z_even & (N != Z)
    P1[m] = 1
    P2[m] = np.abs(N[m] - Z[m]) / 2 - 1
    P3[m] = 1

    return pd.Series(P1, index=idx), pd.Series(P2, index=idx), pd.Series(P3, index=idx)
