"""
nuclear_su4.processing.merge
=============================
Merge the AME2020 and NUBASE2020 DataFrames into a single table.
"""

from __future__ import annotations

import pandas as pd


# Columns brought in from NUBASE (all others kept from AME)
_NUBASE_COLS = [
    "Z", "N", "A",
    "half_life_s", "half_life_unc_str", "spin_parity",
    "alpha_branch_pct", "beta_minus_branch_pct",
    "beta_plus_branch_pct", "EC_branch_pct", "SF_branch_pct",
    "alpha_partial_hl_s", "beta_minus_partial_hl_s", "beta_plus_partial_hl_s",
]


def add_ame_source(df: pd.DataFrame, ame2016_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stamp each row of the merged AME2020 DataFrame with its data origin.

    A nucleus is labelled ``"AME2016"`` if it was **experimentally measured**
    (non-extrapolated) in AME2016.  Everything else — nuclei first measured
    between 2016 and 2020, or nuclei that were extrapolated in 2016 but are
    now experimentally known — is labelled ``"AME2020"``.

    This split is used for the extrapolation benchmark: train on AME2016,
    predict the AME2020 nuclei, and compare against the newly measured masses.

    The ``ame_source`` and ``extrapolated`` columns are **orthogonal**:

    +--------------+----------------+---------------------------------------+-----------------------------------+
    |``ame_source``|``extrapolated``| Meaning                               | Typical use                       |
    +==============+================+=======================================+===================================+
    | ``"AME2016"``| ``False``      | Experimentally measured in AME2016    | Train / test split (80 / 20)      |
    +--------------+----------------+---------------------------------------+-----------------------------------+
    | ``"AME2016"``| ``True``       | Extrapolated in AME2016               | Dropped by ``keep_extrapolated``  |
    +--------------+----------------+---------------------------------------+-----------------------------------+
    | ``"AME2020"``| ``False``      | Newly measured between 2016 and 2020  | Extrapolation validation set      |
    +--------------+----------------+---------------------------------------+-----------------------------------+
    | ``"AME2020"``| ``True``       | Still extrapolated in AME2020         | Dropped by ``keep_extrapolated``  |
    +--------------+----------------+---------------------------------------+-----------------------------------+

    Parameters
    ----------
    df        : Merged AME2020 + NUBASE2020 DataFrame (output of :func:`build`).
    ame2016_df: Parsed AME2016 DataFrame (output of ``parse_ame2016``).

    Returns
    -------
    pd.DataFrame  — copy of *df* with a new ``"ame_source"`` column.
    """
    # Set of (Z, N) that were experimentally known in AME2016
    exp_2016 = set(
        zip(
            ame2016_df.loc[~ame2016_df["extrapolated"], "Z"],
            ame2016_df.loc[~ame2016_df["extrapolated"], "N"],
        )
    )

    df = df.copy()
    df["ame_source"] = df.apply(
        lambda r: "AME2016" if (r["Z"], r["N"]) in exp_2016 else "AME2020",
        axis=1,
    )

    n_2016 = (df["ame_source"] == "AME2016").sum()
    n_2020 = (df["ame_source"] == "AME2020").sum()
    # Break down AME2020 into measured vs still-extrapolated for transparency
    n_2020_exp   = ((df["ame_source"] == "AME2020") & ~df["extrapolated"]).sum()
    n_2020_extrap = ((df["ame_source"] == "AME2020") &  df["extrapolated"]).sum()
    print(f"  ame_source:  {n_2016} AME2016  |  {n_2020} AME2020 "
          f"({n_2020_exp} measured, {n_2020_extrap} extrapolated)")
    return df


def build(ame_df: pd.DataFrame, nubase_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join AME2020 onto NUBASE2020 on (Z, N, A).

    All AME nuclides are kept; NUBASE columns are NaN where no match
    exists (rare — typically only very exotic extrapolated nuclides).

    Parameters
    ----------
    ame_df    : Output of :func:`parsers.ame2020.parse`.
    nubase_df : Output of :func:`parsers.nubase2020.parse`.

    Returns
    -------
    pd.DataFrame
        Merged table, sorted by (A, Z).
    """
    missing = [c for c in _NUBASE_COLS if c not in nubase_df.columns]
    if missing:
        raise ValueError(f"NUBASE DataFrame missing expected columns: {missing}")

    df = ame_df.merge(nubase_df[_NUBASE_COLS], on=["Z", "N", "A"], how="left")
    df = df.sort_values(["A", "Z"]).reset_index(drop=True)
    print(f"  Merged:      {len(df):>5d} rows  ×  {len(df.columns)} columns")
    return df
