"""
build.py  —  One-shot setup script for symass.

Steps
-----
1. Install all required Python packages.
2. Download AME2020 and NUBASE2020 raw data files.
3. Parse and merge the two tables.
4. Annotate with SU(4) x SU(3) symmetry columns.
5. Save the final DataFrame (CSV + pickle) to data/.

Run once before any modelling work:
    python src/build.py
"""

import shutil
import subprocess
import sys
import argparse
from pathlib import Path

# ── 1. Install dependencies ────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "requests",
    "torch",
    "matplotlib",
    "scikit-learn",   # SVR, GridSearchCV, StandardScaler, TransformedTargetRegressor
    "scipy",          # Spearman correlation, temperature scaling (minimize_scalar)
    "shap",           # SHAP feature importance plots
    "masstable",      # Theoretical nuclear mass tables (HFB26, FRDM95, WS3, etc.)
]

def _install_minepy() -> None:
    """
    Install minepy, trying progressively older Python interpreters until one
    succeeds.  minepy's Cython code does not compile against Python 3.13+'s
    overhauled C API, so we fall back to python3.12 … python3.9 if the
    current interpreter is too new.
    """
    # Build candidate list: current interpreter first, then older ones from PATH.
    candidates = [sys.executable] + [
        c for minor in range(12, 8, -1)
        if (c := shutil.which(f"python3.{minor}")) is not None
    ]
    # Deduplicate while preserving order (current interpreter may already be 3.9)
    seen: set[str] = set()
    unique_candidates = []
    for c in candidates:
        resolved = str(Path(c).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(c)

    for py in unique_candidates:
        result = subprocess.run(
            [py, "-m", "pip", "install", "--quiet", "--no-build-isolation", "minepy"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            if py != sys.executable:
                print(f"OK  (installed into {py})")
            else:
                print("OK")
            return

    # All candidates failed — print a clear manual-install instruction.
    print(
        "SKIPPED — minepy requires Python ≤ 3.12 and none was found on PATH.\n"
        "  Install manually with the Python your Jupyter kernel uses, e.g.:\n"
        "    python3.9 -m pip install minepy --no-build-isolation\n"
        "  MIC analysis in Feature_analysis.ipynb will not work until then."
    )


def install_packages(packages: list[str]) -> None:
    _section("1 · Installing dependencies")
    for pkg in packages:
        print(f"  Installing {pkg} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"FAILED\n{result.stderr.strip()}")

    # ── minepy: requires Python ≤ 3.12 ────────────────────────────────────────
    # minepy's Cython extension uses C API internals (ob_digit, curexc_traceback,
    # PyArray_Descr.subarray, ...) that were removed/changed in Python 3.13.
    # Strategy: try sys.executable first; if it fails (e.g. Python 3.13+), fall
    # back through python3.12 → 3.11 → 3.10 → 3.9, whichever is on PATH first.
    print("  Installing minepy ...", end=" ", flush=True)
    _install_minepy()


# ── 2-5. Data pipeline ─────────────────────────────────────────────────────────

def main(data_dir: Path) -> None:
    _banner("symass  --  Build & Setup")

    install_packages(REQUIRED_PACKAGES)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import symass

    # ── 2. Download ────────────────────────────────────────────────────────────
    _section("2 · Downloading raw data files")
    ame2020_text = symass.fetch_ame2020(data_dir)
    ame2016_text = symass.fetch_ame2016(data_dir)
    nubase_text  = symass.fetch_nubase2020(data_dir)

    # ── 3. Parse ───────────────────────────────────────────────────────────────
    _section("3 · Parsing")
    ame2020_df = symass.parse_ame2020(ame2020_text)
    ame2016_df = symass.parse_ame2016(ame2016_text)
    nubase_df  = symass.parse_nubase2020(nubase_text)

    # ── 4. Merge AME2020 + NUBASE2020 ─────────────────────────────────────────
    _section("4 · Merging AME2020 + NUBASE2020")
    df = symass.merge_tables(ame2020_df, nubase_df)

    # ── 5. Stamp AME source (2016 vs 2020) ────────────────────────────────────
    _section("5 · Stamping AME source column")
    df = symass.add_ame_source(df, ame2016_df)

    # ── 6. Add SU(4) x SU(3) columns ──────────────────────────────────────────
    _section("6 · Computing SU(4) x SU(3) symmetry labels")
    df = symass.add_su4_su3_columns(df)

    # ── 7. Save ────────────────────────────────────────────────────────────────
    _section("7 · Saving dataset")
    symass.save_dataframe(df, data_dir)

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    _section("8 · Summary")
    _summary(df)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    line = "=" * 65
    print(f"\n{line}\n  {title}\n{line}")


def _section(title: str) -> None:
    print(f"\n-- {title} {'-' * (60 - len(title))}")


def _summary(df) -> None:
    import numpy as np

    print(f"  Total nuclides         : {len(df)}")
    print(f"  Experimental masses    : {(~df['extrapolated']).sum()}")
    print(f"  Extrapolated masses    : {df['extrapolated'].sum()}")
    print(f"  Stable  (T1/2 = inf)   : {np.isinf(df['half_life_s']).sum()}")
    print(f"  With half-life data    : {df['half_life_s'].notna().sum()}")
    print(f"  With alpha-branch data : {df['alpha_branch_pct'].notna().sum()}")
    print(f"  With beta-branch data  : {df['beta_minus_branch_pct'].notna().sum()}")
    print(f"  N = Z  nuclei          : {(df['NZ'] == 0).sum()}")
    print(f"  Isobaric chains (A)    : {df['A'].nunique()}")
    print(f"\n  Columns:")
    for col in df.columns:
        print(f"    {col}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-shot setup: install deps, download data, compute symmetry labels, save."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Directory for raw data cache and output files (default: ./data)",
    )
    args = parser.parse_args()
    main(args.data_dir)
