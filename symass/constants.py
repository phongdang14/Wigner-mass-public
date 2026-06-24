"""
Physical constants (CODATA 2018) and AME/NUBASE data source URLs.
"""

# ── Unit conversions ───────────────────────────────────────────────────────────
U_TO_KEV: float = 931_494.10242   # 1 u  →  keV / c²
U_TO_MEV: float = 931.494_10242   # 1 u  →  MeV / c²
KEV_TO_U: float = 1.0 / U_TO_KEV
MEV_TO_U: float = 1.0 / U_TO_MEV

# ── Particle masses [u] ────────────────────────────────────────────────────────
M_PROTON:   float = 1.007_276_466_621   # proton  mass [u]
M_NEUTRON:  float = 1.008_664_915_95    # neutron mass [u]
M_ELECTRON: float = 0.000_548_579_909_07

# ── Shell closures ────────────────────────────────────────────────────────────
# Empirical magic numbers
SHELLS_MAGIC: list[int] = [2, 8, 20, 28, 50, 82, 126]

# Harmonic-oscillator shell closures (cumulative nucleon counts).
# Each major shell η holds (η+1)(η+2) nucleons (spatial deg. × spin 2).
# Cumulative sums: 2, 8, 20, 40, 70, 112, 168, 240, 330, 440
_HO_DEG: list[int] = [(i + 1) * (i + 2) for i in range(10)]
SHELLS_HO: list[int] = [sum(_HO_DEG[: k + 1]) for k in range(len(_HO_DEG))]
# = [2, 8, 20, 40, 70, 112, 168, 240, 330, 440]

# ── Data source URLs ───────────────────────────────────────────────────────────
AME2020_URL    = "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"
AME2016_URL    = "https://www-nds.iaea.org/amdc/ame2016/mass16.txt"
NUBASE2020_URL = "https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt"

# Local cache file names (relative to DATA_DIR set in io.py)
AME2020_CACHE    = "mass_1.mas20"
AME2016_CACHE    = "mass16.txt"
NUBASE2020_CACHE = "nubase_4.mas20"
OUTPUT_CSV       = "nuclear_data_AME2020.csv"
OUTPUT_PKL       = "nuclear_data_AME2020.pkl"
