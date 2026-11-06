# DT-BO: Decision-Tree-Guided Feasibility-Aware Bayesian Optimization for Materials Discovery

Code and data for the paper *Decision-Tree-Guided Feasibility-Aware Bayesian Optimization for Materials Discovery* (M. T. Ehsan and M. Sharma Priyadarshini).

![DT-BO graphical abstract](graphical_abstract_v1.png)

In materials discovery, many candidates turn out to be unstable or fail to form, and this is usually only known after the experiment or simulation has been run. DT-BO learns these feasibility constraints as it goes:

- **Learning the feasible region:** a decision tree learns the feasible region from the feasible/infeasible outcomes of the candidates evaluated so far, and is refitted every 5 iterations.
- **Choosing the next candidate:** a Gaussian process (GP) models the target property, and the next candidate is chosen with Log Expected Improvement.
- **The α rule:** with probability α, the search is restricted to the region the tree predicts to be feasible. Otherwise it covers all candidates, which lets the tree correct its own mistakes.
- **Readable rules:** the fitted tree gives explicit if–then feasibility rules on the input features.

This repository runs DT-BO and conventional ("Normal") BO on the three case studies in the paper, and SCBO on the synthetic one. It also includes the additional two-corner synthetic dataset from the Supplementary Material, with all three methods.

## Repository structure

```
Github_repo/
├── README.md
├── requirements.txt
├── graphical_abstract_v1.png
├── synthetic_dataset/
│   ├── data/7peaks_n2000.csv                 # 2,007 candidates
│   ├── dt_bo_synthetic.py                    # DT-BO
│   ├── normal_bo_synthetic.py                # Normal BO
│   └── scbo_synthetic.py                     # SCBO (one seed per call)
├── additional_synthetic_dataset/             # two-corner function (Supplementary Material)
│   ├── data/2corners_n3000.csv               # 3,008 candidates
│   ├── dt_bo_synthetic.py                    # DT-BO
│   ├── normal_bo_synthetic.py                # Normal BO
│   └── scbo_synthetic.py                     # SCBO (one seed per call)
├── hea/
│   ├── data/dataset.xlsx                     # 601 alloys
│   ├── hea_dt_bo.py                          # DT-BO
│   └── hea_normal_bo.py                      # Normal BO
└── perovskite/
    ├── data/Anubis_HOIP_DT_BO_with_descriptors.xlsx   # 1,276 compositions
    ├── perovskite_dt_bo.py                   # DT-BO
    └── perovskite_normal_bo.py               # Normal BO
```

## Installation

You need Python 3.10 or newer:

```bash
python -m venv dtbo_env
source dtbo_env/bin/activate        # Windows: dtbo_env\Scripts\activate
pip install -r requirements.txt
```

The tests were run with Python 3.13, torch 2.15, BoTorch 0.18.1, GPyTorch 1.15.2, scikit-learn 1.9.1, NumPy 2.5.4 and pandas 3.1.1. The paper's runs used Python 3.14 with the newest versions of these packages at the time.

## Case studies

| Case | Data file | Candidates | Feasible | Inputs | Objective |
|---|---|---|---|---|---|
| Seven-peak synthetic | `synthetic_dataset/data/7peaks_n2000.csv` | 2,007 | 895 (feasible if $g_1 \ge g_2$) | $x_1$, $x_2$ | Maximize $f = g_1 + g_2$ (optimum 0.5) |
| High-entropy alloys (HEA) | `hea/data/dataset.xlsx` | 601 | 24 (phase = SS) | 11 composition-derived descriptors | Maximize bulk modulus (optimum 160.75 GPa) |
| Hybrid perovskites (HOIP) | `perovskite/data/Anubis_HOIP_DT_BO_with_descriptors.xlsx` | 1,276 | 111 (stable) | 14 A-, B- and X-site descriptors | Minimize $\lvert E_g - 1.25\ \text{eV} \rvert$ (optimum 0) |
| Two-corner synthetic (Supplementary) | `additional_synthetic_dataset/data/2corners_n3000.csv` | 3,008 | 1,068 (feasible if $g_1 \ge g_2$) | $x_1$, $x_2$ | Maximize $f = g_1 + g_2$ (optimum 0.5) |

**The seven-peak function:**

$$G_{a,b}(x_1,x_2) = \exp[-((x_1-a)^2 + (x_2-b)^2)]$$

$$g_1 = 0.5G_{5,5} + 0.3G_{2.5,1.5} + 0.4G_{8,8} + 0.35G_{8,0}, \qquad g_2 = 0.8G_{-1.2,-1.2} + 0.6G_{-6,-7} + 0.7G_{-7,3}$$

The candidates are the first 2,000 points of a scrambled Sobol sequence on $[-10, 10]^2$, plus the seven peak centres, so the optimum is in the set.

**The two-corner function (additional dataset):**

$$h(x_1,x_2) = 0.02\left[(x_2 - x_1)^2 - 8^2\right]$$

$$g_1 = 0.5G_{-6,6} + 0.35G_{-8.5,2.5} + 0.3G_{-3,8.5} + 0.4G_{6,-6} + 0.3G_{8.5,-2.5} + 0.25G_{3,-8.5} + h, \qquad g_2 = 0.8G_{2,3} + 0.6G_{-3,-4} - h$$

The $h$ terms cancel in $f = g_1 + g_2$, so they only affect feasibility. They place the feasibility boundary close to the lines $x_2 - x_1 = \pm 8$, which splits the feasible space into two separate corners divided by an infeasible diagonal band:
- **Infeasible band:** holds the global maximum, 0.8 at (2, 3).
- **Upper-left corner:** holds the feasible maximum, 0.5 at (−6, 6).
- **Lower-right corner:** has a competing local peak, 0.4 at (6, −6).

The candidates are the first 3,000 points of the same kind of Sobol sequence, plus the eight peak centres.

## Running the experiments

Run each command from its case folder. With no extra options, every script runs the paper's setup: 50 seeds (0–49) × 300 BO iterations.

### Seven-peak synthetic

```bash
cd synthetic_dataset

# DT-BO
python dt_bo_synthetic.py --dataset data/7peaks_n2000.csv --output-dir results/dt_bo

# Normal BO
python normal_bo_synthetic.py --dataset data/7peaks_n2000.csv --output-dir results/normal_bo

# SCBO: one seed per call
for seed in $(seq 0 49); do
    python scbo_synthetic.py --problem 7peaks --dataset data/7peaks_n2000.csv \
        --seed "$seed" --space discrete --init dtbo --output-dir results/scbo
done
```

On Windows PowerShell, run the SCBO loop as:

```powershell
foreach ($seed in 0..49) { python scbo_synthetic.py --problem 7peaks --dataset data/7peaks_n2000.csv --seed $seed --space discrete --init dtbo --output-dir results/scbo }
```

**SCBO options:**
- **`--space discrete`:** SCBO chooses only from unevaluated candidates in the dataset, like DT-BO and Normal BO. Without it, SCBO searches the continuous space.
- **`--init dtbo`:** SCBO starts from the same 10 initial points as the other two methods.
- **Finished seeds:** these are skipped unless you pass `--overwrite`.

### High-entropy alloys

```bash
cd hea
python hea_dt_bo.py --dataset data/dataset.xlsx --output-dir results/dt_bo
python hea_normal_bo.py --dataset data/dataset.xlsx --output-dir results/normal_bo
```

### Hybrid perovskites

```bash
cd perovskite
python perovskite_dt_bo.py --dataset data/Anubis_HOIP_DT_BO_with_descriptors.xlsx --output-dir results/dt_bo
python perovskite_normal_bo.py --dataset data/Anubis_HOIP_DT_BO_with_descriptors.xlsx --output-dir results/normal_bo
```

### Two-corner synthetic (additional dataset)

```bash
cd additional_synthetic_dataset

# DT-BO
python dt_bo_synthetic.py --dataset data/2corners_n3000.csv --output-dir results/dt_bo

# Normal BO
python normal_bo_synthetic.py --dataset data/2corners_n3000.csv --output-dir results/normal_bo

# SCBO: one seed per call
for seed in $(seq 0 49); do
    python scbo_synthetic.py --problem 2corners --dataset data/2corners_n3000.csv \
        --seed "$seed" --space discrete --init dtbo --output-dir results/scbo
done
```

On Windows PowerShell, run the SCBO loop as:

```powershell
foreach ($seed in 0..49) { python scbo_synthetic.py --problem 2corners --dataset data/2corners_n3000.csv --seed $seed --space discrete --init dtbo --output-dir results/scbo }
```

These are the same three scripts as in `synthetic_dataset/`. The SCBO script needs `--problem 2corners` to select the two-corner function, and it checks that the CSV's `g1` and `g2` columns match that function.

### Quick test and other options

Each run uses `--seeds 1 --iterations 5` for a quick test, or `--seed 0 --iterations 5` for SCBO. Run any script with `--help` to see all options:
- **Seeds:** `--seed-start` changes the first seed.
- **DT-BO settings:** `--alpha`, the tree settings and the refit interval can all be changed.
- **Hardware:** `--device cpu` or `cuda`, in the DT-BO and Normal BO scripts.

On the authors' cluster, using 8 CPU cores, the seven-peak runs took about 3.7 h for DT-BO, 0.9 h for Normal BO and 1.4 h for SCBO, each covering 50 seeds.

## Method settings

| Setting | Value |
|---|---|
| Initial points | 5 feasible + 5 infeasible per seed, drawn with `numpy.random.default_rng(seed)`; the same points are used for every method |
| Budget | 300 BO iterations per seed, 50 seeds |
| Surrogate | BoTorch `SingleTaskGP` with standardized outputs; inputs scaled to [0, 1]; hyperparameters by exact marginal likelihood |
| Acquisition | Log Expected Improvement, evaluated on every candidate in the search set |
| DT-BO search | With probability α = 0.9, only the tree's predicted-feasible candidates; otherwise all remaining candidates; all candidates if none are predicted feasible |
| DT-BO GP training data | Evaluated feasible candidates only |
| Normal BO GP training data | All evaluated candidates with an objective value (only feasible ones for HOIP, because infeasible compositions have no band gap) |
| Decision tree | scikit-learn `DecisionTreeClassifier` with balanced class weights, refitted every 5 iterations on the evaluated candidates. Max depth / min samples split / min samples leaf: 6 / 5 / 2 for synthetic and HEA, 8 / 4 / 2 for HOIP |
| SCBO | BoTorch's SCBO tutorial implementation, adapted to the discrete candidate set; constraint $c = g_2 - g_1 \le 0$; one evaluation per iteration; trust-region restarts count toward the budget |

## Outputs

Every row is one BO iteration of one seed, and each file has `Seed` and `BO_Iteration` columns.

| Script | Files | Best-so-far column |
|---|---|---|
| `dt_bo_synthetic.py` | `DT_BO_All_Seeds_Iterations.csv`, `DT_Classifier_Refit_Metrics.csv` | `Best_Stable_Objective` |
| `normal_bo_synthetic.py` | `Normal_BO_All_Seeds_Iterations.csv` | `Best_Stable_Objective` |
| `scbo_synthetic.py` | `scbo_discrete/seed_XXX.csv` (BO iterations), `scbo_discrete/full_log_seed_XXX.csv` (all evaluations) | `Best_Stable_Objective` |
| `hea_dt_bo.py` | `DT_BO_All_Seeds_Iterations.csv`, `DT_Classifier_Refit_Metrics.csv` | `Best_SS_BulkModulus` (in Pa) |
| `hea_normal_bo.py` | `Normal_BO_All_Seeds_Iterations.csv` | `Best_SS_BulkModulus` (in Pa) |
| `perovskite_dt_bo.py` | `DT_BO_All_Seeds_Iterations.csv`, `DT_Classifier_Refit_Metrics.csv` | `Best_Feasible_f1` (in eV) |
| `perovskite_normal_bo.py` | `Normal_BO_All_Seeds_Iterations.csv` | `Best_Feasible_f1` (in eV) |

`DT_Classifier_Refit_Metrics.csv` scores each refitted tree against the full set of labels. These scores, including the F1 score, are for diagnosis only. They are never used for training or for choosing candidates.

## Reproducibility notes

- **Synthetic scripts:** these are the exact scripts that produced the paper's seven-peak results and the Supplementary two-corner results.
- **HEA and HOIP scripts:** the paper ran DT-BO and Normal BO together in one script per dataset. Here they're split into two scripts, with the plotting code removed. The split scripts give the same results as the originals; this was checked on 2 seeds × 15 iterations.
- **Exact numbers:** results depend on library versions, hardware and thread count, so a rerun may not match the paper exactly. Expect small differences in the selected candidates and similar overall results.

## Data sources

Please cite the original sources if you use these datasets:

- **HEA dataset:** Z. Zhou, Y. Zhou, Q. He, Z. Ding, F. Li and Y. Yang, *npj Computational Materials*, 2019, **5**, 128.
- **HOIP dataset:** S. Körbel, M. A. L. Marques and S. Botti, *Journal of Materials Chemistry A*, 2018, **6**, 6463–6475. It was used as a constrained-optimization benchmark in R. J. Hickman et al., *Digital Discovery*, 2025, **4**, 2104–2122 (Anubis).

## Baseline method

SCBO was proposed by D. Eriksson and M. Poloczek, *Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2021, pp. 730–738. The implementation here is adapted from the BoTorch SCBO tutorial.

## Citation

If you use this code, please cite the paper (details will be added after publication):

```bibtex
@article{ehsan_dtbo,
  title   = {Decision-Tree-Guided Feasibility-Aware Bayesian Optimization for Materials Discovery},
  author  = {Ehsan, Mashaekh Tausif and Sharma Priyadarshini, Maitreyee},
  journal = {},
  year    = {}
}
```

## Contact

Maitreyee Sharma Priyadarshini, Kevin T. Crofton Department of Aerospace and Ocean Engineering, Virginia Tech (msharmap@vt.edu).
