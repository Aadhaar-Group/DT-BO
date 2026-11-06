#!/usr/bin/env python3
"""SCBO (Scalable Constrained Bayesian Optimization) on the synthetic case study.

Adapted from the BoTorch tutorial
  tutorials/scalable_constrained_bo/scalable_constrained_bo.ipynb
  (Eriksson & Poloczek, AISTATS 2021)
The tutorial optimizes 10-D Ackley with two constraints. Here the problem is
one of the synthetic case studies, chosen with --problem (G(a,b) = exp(-((x1-a)^2 + (x2-b)^2)),
h(x1,x2) = K ((x2-x1)^2 - W^2), f = g1 + g2 maximized on [-10,10]^2, feasible <=> g1 >= g2):

  4peaks    g1 = 0.5 G(2.5,1.5) + 0.3 G(5,5)
            g2 = 0.8 G(-1.2,-1.2) + 0.6 G(-6,-7)                       optimum 0.5 at (2.5, 1.5)
  7peaks    g1 = 0.5 G(5,5) + 0.3 G(2.5,1.5) + 0.4 G(8,8) + 0.35 G(8,0)
            g2 = 0.8 G(-1.2,-1.2) + 0.6 G(-6,-7) + 0.7 G(-7,3)        optimum 0.5 at (5, 5)
  2corners  g1 = 0.5 G(-6,6) + 0.35 G(-8.5,2.5) + 0.3 G(-3,8.5)
                 + 0.4 G(6,-6) + 0.3 G(8.5,-2.5) + 0.25 G(3,-8.5) + h
            g2 = 0.8 G(2,3) + 0.6 G(-3,-4) - h   (W = 8, K = 0.02)    optimum 0.5 at (-6, 6)

Dataset-size analysis version (marked "CHANGED (dataset size)"): scbo_synthetic_7peaks.py
with the problem chosen by --problem (g1/g2 are checked against the CSV's g1/g2 columns at
start) and the candidate CSV given by --dataset (datasets/<problem>_n<N>.csv). Run with
--space discrete. --init dtbo draws 5 stable + 5 unstable rows exactly as
nature_bulk_modulus_dt_bo_hpc.py / dt_bo_synthetic.py / normal_bo_synthetic.py
(np.random.default_rng(seed): 5 feasible, 5 infeasible, shuffled), so all three methods
start from the same 10 points per seed.

SCBO needs constraints of the form c(x) <= 0, so the single constraint is
  c(x) = g2(x) - g1(x) <= 0.
Like the tutorial, every evaluation returns both f(x) and c(x), and all points
(feasible or not) are used to fit the objective and constraint GPs.

What is kept from the tutorial (unchanged): ScboState, update_tr_length,
get_best_index_for_batch, update_state, generate_batch with
ConstrainedMaxPosteriorSampling, get_fitted_model (Matern-5/2 ARD, noise in
[1e-8, 1e-3], lengthscale in [0.005, 4]).

What is changed for a benchmark run (marked CHANGED):
  * problem: 2-D synthetic functions instead of 10-D Ackley
  * fixed budget (10 initial + 300 evaluations, like DT-BO) instead of
    "run until the trust region collapses"; when it collapses, TuRBO restarts
    with a new trust region from fresh Sobol points, as in the SCBO paper
  * batch_size = 1 (one evaluation per iteration, like DT-BO; tutorial: 4)
  * seeds: one seed per SLURM array task
  * initial points: the same 10 points as synthetic_dt_bo_arc_*.py for that
    seed (--init dtbo, needs the CSV), or the tutorial's Sobol points (--init sobol)
  * --space discrete: Thompson-sampling candidates are the unmeasured CSV points
    inside the trust region (the same candidate CSV as DT-BO) instead of
    continuous perturbations of the trust-region centre
  * results written to scbo_results/seed_XXX.csv

Usage
  python scbo_synthetic.py --problem 7peaks --dataset ../datasets/7peaks_n2000.csv --seed 0 --space discrete
  python scbo_synthetic.py --problem 7peaks --dataset ../datasets/7peaks_n2000.csv --seed 0 --space discrete --iterations 20
"""

import argparse
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import unnormalize, normalize

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}


# ---------------------------------------------------------------------------
# CHANGED: the synthetic case study (paper Eq. 1-4) instead of Ackley + c1, c2
# ---------------------------------------------------------------------------
BOUNDS = torch.tensor([[-10.0, -10.0], [10.0, 10.0]], **tkwargs)
DIM = 2
# CHANGED (dataset size): the three synthetic problems; (height, x1 centre, x2 centre),
# boundary term h = K ((x2-x1)^2 - W^2) (K = 0: no boundary term), best feasible point
PROBLEMS = {
    "4peaks": dict(G1=[(0.5, 2.5, 1.5), (0.3, 5.0, 5.0)],
                   G2=[(0.8, -1.2, -1.2), (0.6, -6.0, -7.0)],
                   W=0.0, K=0.0, OPT=(2.5, 1.5)),
    "7peaks": dict(G1=[(0.5, 5.0, 5.0), (0.3, 2.5, 1.5), (0.4, 8.0, 8.0), (0.35, 8.0, 0.0)],
                   G2=[(0.8, -1.2, -1.2), (0.6, -6.0, -7.0), (0.7, -7.0, 3.0)],
                   W=0.0, K=0.0, OPT=(5.0, 5.0)),
    "2corners": dict(G1=[(0.5, -6.0, 6.0), (0.35, -8.5, 2.5), (0.3, -3.0, 8.5),
                         (0.4, 6.0, -6.0), (0.3, 8.5, -2.5), (0.25, 3.0, -8.5)],
                     G2=[(0.8, 2.0, 3.0), (0.6, -3.0, -4.0)],
                     W=8.0, K=0.02, OPT=(-6.0, 6.0)),
}
G1_PEAKS = G2_PEAKS = TRUE_OPT_X = TRUE_OPT_VALUE = None   # set by set_problem()
W = K = 0.0


def _gaussian_sum(peaks, X: Tensor) -> Tensor:
    x1, x2 = X[..., 0], X[..., 1]
    return sum(h * torch.exp(-((x1 - a) ** 2 + (x2 - b) ** 2)) for h, a, b in peaks)


def h(X: Tensor) -> Tensor:
    # CHANGED (dataset size): boundary term of the two-corner problem (zero when K = 0)
    if K == 0.0:
        return torch.zeros_like(X[..., 0])
    return K * ((X[..., 1] - X[..., 0]) ** 2 - W ** 2)


def g1(X: Tensor) -> Tensor:
    return _gaussian_sum(G1_PEAKS, X) + h(X)


def g2(X: Tensor) -> Tensor:
    return _gaussian_sum(G2_PEAKS, X) - h(X)


def f(X: Tensor) -> Tensor:
    """Objective, Eq. (1): f = g1 + g2 (maximized)."""
    return g1(X) + g2(X)


def c(X: Tensor) -> Tensor:
    """Constraint in SCBO form, c(x) <= 0  <=>  g1 >= g2 (Class 1, Eq. 4)."""
    return g2(X) - g1(X)


def set_problem(name):
    # CHANGED (dataset size): select the synthetic problem (module-level settings)
    global G1_PEAKS, G2_PEAKS, W, K, TRUE_OPT_X, TRUE_OPT_VALUE
    p = PROBLEMS[name]
    G1_PEAKS, G2_PEAKS, W, K, TRUE_OPT_X = p["G1"], p["G2"], p["W"], p["K"], p["OPT"]
    TRUE_OPT_VALUE = float(f(torch.tensor([TRUE_OPT_X], **tkwargs)))


# As in the tutorial: the optimizer works on [0,1]^d, evaluations unnormalize
def eval_objective(x):
    return f(unnormalize(x, BOUNDS))


def eval_c1(x):
    return c(unnormalize(x, BOUNDS))


# ---------------------------------------------------------------------------
# Tutorial code (unchanged): trust-region state and its update
# ---------------------------------------------------------------------------
@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = None   # CHANGED: one constraint, set below
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))
        if self.best_constraint_values is None:
            self.best_constraint_values = torch.ones(1, **tkwargs) * torch.inf


def update_tr_length(state: ScboState):
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_state(state, Y_next, C_next):
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    state = update_tr_length(state)
    return state


def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


def generate_batch(state, model, X, Y, C, batch_size, n_candidates, constraint_model,
                   sobol: SobolEngine, pool=None):
    """Tutorial generate_batch. CHANGED: if `pool` (normalized CSV points not yet
    measured) is given, the Thompson-sampling candidates are the pool points
    inside the trust region instead of continuous perturbations."""
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Create the TR bounds
    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]
    if pool is None:
        # Thompson Sampling w/ Constraints (SCBO), as in the tutorial
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]
    else:
        inside = ((pool >= tr_lb) & (pool <= tr_ub)).all(dim=-1)
        X_cand = pool[inside]
        if X_cand.shape[0] < batch_size:   # trust region holds no unmeasured point
            d = ((pool - x_center) ** 2).sum(-1)
            X_cand = pool[torch.argsort(d)[: max(n_candidates, batch_size)]]
        elif X_cand.shape[0] > n_candidates:  # same number of TS candidates as the tutorial
            X_cand = X_cand[torch.randperm(X_cand.shape[0], device=device)[:n_candidates]]

    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


max_cholesky_size = float("inf")  # Always use Cholesky


def get_fitted_model(X, Y):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=X.shape[-1], lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(X, Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError:
            # CHANGED: far from the peaks f and c are ~1e-80 everywhere, so the
            # data can be (numerically) constant and hyperparameter fitting fails.
            # Keep the model with its default hyperparameters for this iteration.
            global N_FIT_FAILURES
            N_FIT_FAILURES += 1
    model.eval()
    return model


N_FIT_FAILURES = 0


# ---------------------------------------------------------------------------
# CHANGED: fixed-budget benchmark loop with restarts, seeds, CSV output
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SCBO on the synthetic case study")
    p.add_argument("--seed", type=int, default=None,
                   help="Default: $SLURM_ARRAY_TASK_ID + --seed-offset")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--iterations", type=int, default=300, help="evaluations after the initial points")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-candidates", type=int, default=2000,
                   help="Thompson-sampling candidates (tutorial: 2000)")
    p.add_argument("--init", choices=["dtbo", "sobol"], default="dtbo",
                   help="dtbo: same 10 initial points (5 stable + 5 unstable) as "
                        "synthetic_dt_bo_arc_*.py (needs --dataset); sobol: tutorial initial design")
    p.add_argument("--space", choices=["continuous", "discrete"], default="continuous",
                   help="continuous: SCBO as in the tutorial; discrete: only the CSV candidates")
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)   # CHANGED (dataset size)
    p.add_argument("--dataset", required=True, help="candidate CSV of that problem")   # CHANGED (dataset size)
    p.add_argument("--output-dir", default="scbo_results")
    p.add_argument("--overwrite", action="store_true", help="rerun seeds whose result file exists")
    return p.parse_args()


def main():
    a = parse_args()
    set_problem(a.problem)   # CHANGED (dataset size)
    if a.seed is None:
        a.seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) + a.seed_offset
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    out_dir = Path(a.output_dir) / (f"scbo_{a.space}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seed_{a.seed:03d}.csv"
    if out_csv.is_file() and not a.overwrite:   # CHANGED: finished seeds are skipped
        print(f"{out_csv} already exists; skipping seed {a.seed} (use --overwrite to rerun)")
        return

    # candidate list (needed for --init dtbo and --space discrete)
    frame = None
    if a.init == "dtbo" or a.space == "discrete":
        if not Path(a.dataset).is_file():
            raise FileNotFoundError(f"{a.dataset} not found (needed for --init dtbo / --space discrete)")
        frame = pd.read_csv(a.dataset)
        X_all = torch.tensor(frame[["x1", "x2"]].to_numpy(), **tkwargs)
        # CHANGED (dataset size): the CSV must come from the same landscape as g1/g2 above
        for col, fn in (("g1", g1), ("g2", g2)):
            if not np.allclose(fn(X_all).cpu().numpy(), frame[col].to_numpy(), rtol=1e-9, atol=1e-15):
                raise ValueError(f"{a.dataset}: column {col} does not match {fn.__name__}() in this script")
        X_all_unit = normalize(X_all, BOUNDS)
        measured = torch.zeros(len(frame), dtype=torch.bool, device=device)

    # ---- initial data --------------------------------------------------------
    if a.init == "dtbo":
        # CHANGED (dataset size): 5 stable + 5 unstable, same draws as make_initial_indices
        # in synthetic_dt_bo_arc_*.py (was 10 random rows)
        if a.n_init != 10:
            raise ValueError("--init dtbo uses 5 stable + 5 unstable points; keep --n-init 10")
        labels = frame["True_Stable"].to_numpy(dtype=int)
        rng = np.random.default_rng(a.seed)
        init_stable = rng.choice(np.where(labels == 1)[0], size=5, replace=False)
        init_unstable = rng.choice(np.where(labels == 0)[0], size=5, replace=False)
        rows = np.concatenate([init_stable, init_unstable]).astype(int)
        rng.shuffle(rows)
        train_X = X_all_unit[torch.as_tensor(rows, device=device)]
    else:
        train_X = get_initial_points(DIM, a.n_init, seed=a.seed)
        rows = None
    if frame is not None and a.space == "discrete" and rows is not None:
        measured[torch.as_tensor(rows, device=device)] = True
    train_Y = torch.tensor([eval_objective(x) for x in train_X], **tkwargs).unsqueeze(-1)
    C1 = torch.tensor([eval_c1(x) for x in train_X], **tkwargs).unsqueeze(-1)

    log = []
    phase = ["init"] * a.n_init
    restart_id = [0] * a.n_init
    tr_len = [float("nan")] * a.n_init

    state = ScboState(DIM, batch_size=a.batch_size)
    sobol = SobolEngine(DIM, scramble=True, seed=a.seed + 1)
    # trust-region data: the current TuRBO run uses only its own points
    tr_X, tr_Y, tr_C = train_X.clone(), train_Y.clone(), C1.clone()
    n_restarts = 0
    budget = a.n_init + a.iterations
    t0 = time.time()

    while train_X.shape[0] < budget:
        if state.restart_triggered:
            # CHANGED: restart TuRBO with a new trust region (SCBO paper);
            # the restart's initial points count toward the budget
            n_restarts += 1
            state = ScboState(DIM, batch_size=a.batch_size)
            n_new = min(a.n_init, budget - train_X.shape[0])
            X_new = get_initial_points(DIM, n_new, seed=a.seed * 1000 + n_restarts)
            if a.space == "discrete":   # snap restart points to unmeasured CSV points
                snapped = []
                for x in X_new:
                    d = ((X_all_unit - x) ** 2).sum(-1)
                    d[measured] = float("inf")
                    j = int(torch.argmin(d)); measured[j] = True
                    snapped.append(X_all_unit[j])
                X_new = torch.stack(snapped)
            Y_new = torch.tensor([eval_objective(x) for x in X_new], **tkwargs).unsqueeze(-1)
            C_new = torch.tensor([eval_c1(x) for x in X_new], **tkwargs).unsqueeze(-1)
            train_X = torch.cat((train_X, X_new)); train_Y = torch.cat((train_Y, Y_new)); C1 = torch.cat((C1, C_new))
            tr_X, tr_Y, tr_C = X_new.clone(), Y_new.clone(), C_new.clone()
            phase += ["restart_init"] * n_new; restart_id += [n_restarts] * n_new; tr_len += [float("nan")] * n_new
            continue

        # Fit GP models for objective and constraint (tutorial)
        model = get_fitted_model(tr_X, tr_Y)
        c1_model = get_fitted_model(tr_X, tr_C)

        pool = None
        if a.space == "discrete":
            pool = X_all_unit[~measured]

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            X_next = generate_batch(
                state=state, model=model, X=tr_X, Y=tr_Y, C=tr_C,
                batch_size=min(a.batch_size, budget - train_X.shape[0]),
                n_candidates=a.n_candidates,
                constraint_model=ModelListGP(c1_model),
                sobol=sobol, pool=pool,
            )
        if a.space == "discrete":
            for x in X_next:
                j = int(torch.argmin(((X_all_unit - x) ** 2).sum(-1)))
                measured[j] = True

        Y_next = torch.tensor([eval_objective(x) for x in X_next], **tkwargs).unsqueeze(-1)
        C_next = torch.tensor([eval_c1(x) for x in X_next], **tkwargs).unsqueeze(-1)

        tr_len += [state.length] * X_next.shape[0]
        state = update_state(state=state, Y_next=Y_next, C_next=C_next)

        # Append all data, feasible or not (tutorial)
        train_X = torch.cat((train_X, X_next)); train_Y = torch.cat((train_Y, Y_next)); C1 = torch.cat((C1, C_next))
        tr_X = torch.cat((tr_X, X_next)); tr_Y = torch.cat((tr_Y, Y_next)); tr_C = torch.cat((tr_C, C_next))
        phase += ["scbo"] * X_next.shape[0]; restart_id += [n_restarts] * X_next.shape[0]

        n = train_X.shape[0]
        if n % 25 == 0 or n == a.n_init + 1:
            feas = (C1 <= 0).squeeze(-1)
            best = train_Y[feas].max().item() if feas.any() else float("nan")
            print(f"seed {a.seed} | {n}/{budget} evals | best feasible {best:.6g} "
                  f"(true {TRUE_OPT_VALUE:.6g}) | TR length {state.length:.3g} | restarts {n_restarts} "
                  f"| {time.time() - t0:.0f}s", flush=True)

    # ---- per-evaluation table ------------------------------------------------
    X_raw = unnormalize(train_X, BOUNDS).cpu().numpy()
    Y = train_Y.squeeze(-1).cpu().numpy()
    C = C1.squeeze(-1).cpu().numpy()
    feas = (C <= 0).astype(int)
    G1 = g1(unnormalize(train_X, BOUNDS)).cpu().numpy()
    G2 = g2(unnormalize(train_X, BOUNDS)).cpu().numpy()
    best_so_far = np.maximum.accumulate(np.where(feas == 1, Y, -np.inf))
    best_so_far[np.isinf(best_so_far)] = np.nan

    if frame is not None:
        # nearest CSV row (exact match in discrete mode / for dtbo initial points)
        Xr = frame[["x1", "x2"]].to_numpy()
        idx = [int(np.argmin(((Xr - x) ** 2).sum(1))) for x in X_raw]
        exact = [bool(np.allclose(Xr[i], x, atol=1e-9)) for i, x in zip(idx, X_raw)]
        ds_index = [int(frame.Dataset_Index[i]) if e else -1 for i, e in zip(idx, exact)]
        gbest = int(frame.loc[(frame.x1 == TRUE_OPT_X[0]) & (frame.x2 == TRUE_OPT_X[1]), "Dataset_Index"].iloc[0]) \
            if ((frame.x1 == TRUE_OPT_X[0]) & (frame.x2 == TRUE_OPT_X[1])).any() else -2
    else:
        ds_index = [-1] * len(Y); gbest = -2

    table = pd.DataFrame({
        "Method": f"SCBO ({a.space})", "Seed": a.seed,
        "Evaluation": np.arange(1, len(Y) + 1),
        "BO_Iteration": np.arange(1, len(Y) + 1) - a.n_init,
        "Phase": phase, "TuRBO_Restart": restart_id, "TR_Length_Before": tr_len,
        "x1": X_raw[:, 0], "x2": X_raw[:, 1], "g1": G1, "g2": G2,
        "Objective": Y, "Constraint_c": C,
        "True_Feasible": feas, "Selected_Objective": Y,
        "Best_Stable_Objective": best_so_far,
        "Dataset_Index": ds_index,
        "Initial_Evaluations": a.n_init,
        "Global_Best_Objective": TRUE_OPT_VALUE,
        "Global_Best_Dataset_Index": gbest,
        "Global_Best_In_Initial": int(gbest in ds_index[: a.n_init]),
        "Regret": TRUE_OPT_VALUE - np.nan_to_num(best_so_far, nan=0.0),
        "Candidate_Space_Size": len(frame) if frame is not None else -1,
        "Init": a.init, "Space": a.space, "Batch_Size": a.batch_size,
    })
    # keep only BO iterations (>=1) in the comparison file, full log alongside
    table.to_csv(out_dir / f"full_log_seed_{a.seed:03d}.csv", index=False)
    table[table.BO_Iteration >= 1].to_csv(out_csv, index=False)

    bo = table[table.BO_Iteration >= 1]
    print(f"\nseed {a.seed} done in {(time.time() - t0) / 60:.1f} min | best feasible "
          f"{bo.Best_Stable_Objective.iloc[-1]:.6g} / {TRUE_OPT_VALUE:.6g} | "
          f"infeasible BO evals {100 * (1 - bo.True_Feasible.mean()):.1f}% | restarts {n_restarts} "
          f"| GP fits kept at defaults {N_FIT_FAILURES}\n"
          f"-> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
