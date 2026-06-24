# Machine Learning Nuclear Mass Models inspired by Emergent Symmetries

By Phong Dang

---

## Background

Advances in machine learning and data science have opened new doors for nuclear physicists to model and predict nuclear masses. At the same time, *ab initio* calculations have revealed the dominance of Wigner's SU(4) and Elliott's SU(3) symmetries in the nuclear force and in the structure of light and intermediate-mass nuclei. The `symass` library brings these threads together, providing the tools to probe the influence of these symmetries on nuclear binding energies (and potentially other observables) across the entire nuclear chart.

---

## Features

The library can supply the following input features and functions thereof:

| Feature | Physical meaning |
|---------|-----------------|
| $Z$ | Proton number  |
| $N$ | Neutron number |
| $A$ | Mass number |
| $\mathcal{C}_2[\mathrm{SU(4)}]$, $\mathcal{C}_3[\mathrm{SU(4)}]$, $\mathcal{C}_4[\mathrm{SU(4)}]$ | Two-body, three-body, four-body spin–isospin polarization forces |
| $T_z = (N-Z)/2$ | Isospin projection |
| $N_{\hbar\omega}$ | Total harmonic-oscillator quanta |
| $\mathcal{C}_2[\mathrm{SU(3)}]$ | Quadrupole deformation |
| $\mathcal{C}_3[\mathrm{SU(3)}]$ | Triaxiality |

The signed cubic Casimir $\mathcal{C}_3[\mathrm{SU(4)}]$ can additionally be split into two nonnegative branches $\mathcal{C}_3^{+}$ and $\mathcal{C}_3^{-}$ (explored in `Feature_analysis.ipynb` and `FINN_SplitC3_4.ipynb`).

---

## Loss functions

Two loss variants are provided:

- **MSE** through `train` function`: mean-squared-error on binding energy
- **MSE + PINN** through `train_pinn` function: MSE plus the Garvey–Kelson 6-body relations (see, e.g., Phys Rev C 106 (2022) L021301)

---

## Code structure

```
symass/                   # Python library
  constants.py            # Physical constants and AME/NUBASE data-source URLs
  ame2020.py              # Parser for the AME2020 / AME2016 mass tables
  nubase2020.py           # Parser for NUBASE2020 (half-lives, decay modes)
  merge.py                # Merge AME + NUBASE tables; stamp AME source (2016 vs 2020)
  su4xsu3.py              # Build SU(4)×SU(3) features: leading irreps and Casimir invariants
  data_manager.py         # Download/fetch, save/load datasets, Z-score Normalizer, feature loaders
  soap.py                 # SOAP optimizer
  training.py             # Training helpers, early stopping, Garvey–Kelson (PINN) physics loss
  validate.py             # Model evaluation utilities (RMSE on train/test/validation)
  plot.py                 # Nuclear-chart heatmaps, prediction/residual, SHAP and correlation plots
src/
  build.py                # One-shot setup: install dependencies, download data and build the dataset
work/                     # Jupyter notebooks
  Feature_analysis.ipynb  # Feature heatmaps, Spearman & MIC correlation analysis
  FINN.ipynb              # Plain feed-forward NN with binding energy as single output
  FINN_SplitC3_4.ipynb    # FINN augmented with the split C₃[SU(4)] features
  GINN.ipynb              # Feed-forward NN with binding energy and uncertainty as outputs
  WINN.ipynb              # Interpretable mass model with (N,Z)-dependent coupling functions
data/                     # AME2016 / AME2020 / NUBASE2020 raw data files set up from src/build.py
```

---

## Getting started

```bash
git clone git@github.com:phongdang14/Wigner-mass-public.git
cd Wigner-mass-public
python3 src/build.py
```

`build.py` installs all dependencies, downloads the AME2016/AME2020/NUBASE2020 raw data, computes the SU(4)×SU(3) symmetry columns, and saves the final dataset to `data/`. This only needs to be run once. After that, open any notebook in `work/`.

---

## Implementation notes

- Optimizer: `SOAP` (second-order, analogous to Shampoo), which can be found at this [link](https://github.com/nikhilvyas/SOAP)  
- Normalisation: Z-score on both X and y (via the `Normalizer` class)

---

## Credits

| Role | Contributor(s) |
|------|----------------|
| Concept & theory | Phong Dang, Xiaoliang Wan, Jerry Draayer, Feng Pan, Tomas Dytrych, David Kekejian |
| Code development | Phong Dang, Daniel Langr |
| Feature engineering | Michela Negro, Evander Espinoza |
| Data & analysis | Phong Dang, Michela Negro |

If you use this library, please cite: _(citation / DOI — to be added)._
