#!/usr/bin/env python3
"""DT-BO on the hybrid organic-inorganic perovskite (HOIP) dataset: minimize f1 = |band gap - 1.25 eV| among stable compositions.

Feasible = stable (111 of 1276 compositions). The GP is trained on observed feasible
candidates only; a decision tree (max depth 8, min samples split 4, min samples leaf 2,
balanced class weights) is refitted every 5 iterations on the observed labels; with
probability alpha = 0.9 the search is restricted to the tree's predicted-feasible candidates,
otherwise all remaining candidates.

Usage:
    python perovskite_dt_bo.py --dataset data/Anubis_HOIP_DT_BO_with_descriptors.xlsx --output-dir results/dt_bo

Outputs: DT_BO_All_Seeds_Iterations.csv, DT_Classifier_Refit_Metrics.csv

Default experiment: 50 seeds (0-49) x 300 BO iterations, 5 feasible + 5 infeasible initial
points per seed. The initial points depend only on the seed, so DT-BO and Normal BO start
from the same points. Results are checkpointed after every seed.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


ID_COLUMNS = ["candidate_id", "molcat", "metal", "halogen"]
LABEL_COLUMN = "feasible"
BANDGAP_COLUMN = "bandgap"
OBJECTIVE_COLUMN = "f1_bandgap_distance"

DESCRIPTOR_COLUMNS = [
    "molcat_scf_e",
    "molcat_homo_e",
    "molcat_lumo_e",
    "molcat_dip_mom_norm",
    "molcat_radius_2d",
    "molcat_mw",
    "metal_electron_affinity",
    "metal_electronegativity",
    "metal_ionization_energy",
    "metal_total_mass",
    "halogen_electron_affinity",
    "halogen_electronegativity",
    "halogen_ionization_energy",
    "halogen_total_mass",
]

DT_CSV_NAME = "DT_BO_All_Seeds_Iterations.csv"
CLASSIFIER_METRICS_CSV_NAME = "DT_Classifier_Refit_Metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DT-BO on the Anubis HOIP perovskite dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Anubis_HOIP_DT_BO_with_descriptors.xlsx"),
        help="Input .xlsx workbook (default: %(default)s).",
    )
    parser.add_argument(
        "--sheet",
        default="DT_BO_Dataset",
        help="Workbook sheet containing the dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dt_bo"),
        help="Directory for CSV and PNG outputs (default: %(default)s).",
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--initial-feasible", type=int, default=5)
    parser.add_argument("--initial-infeasible", type=int, default=5)
    parser.add_argument("--target-bandgap", type=float, default=1.25)
    parser.add_argument("--tree-max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--refit-every", type=int, default=5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.90,
        help="Probability DT-BO searches only predicted-feasible candidates.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device (default: auto).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("seeds", "iterations", "initial_feasible", "initial_infeasible"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start cannot be negative.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.refit_every < 0:
        raise ValueError("--refit-every cannot be negative.")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_scalar(value):
    """Convert NumPy scalars to CSV-friendly native Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def binary_metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    prefix: str,
) -> dict:
    """Return binary-classification metrics without changing model state."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        names = [
            "Samples", "Accuracy", "Balanced_Accuracy", "Precision_Positive",
            "Recall_Positive", "F1_Positive", "Specificity", "ROC_AUC",
            "TN", "FP", "FN", "TP",
        ]
        return {f"{prefix}_{name}": np.nan for name in names}

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    roc_auc = (
        float(roc_auc_score(y_true, y_score))
        if np.unique(y_true).size == 2
        else np.nan
    )
    return {
        f"{prefix}_Samples": int(y_true.size),
        f"{prefix}_Accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_Balanced_Accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        f"{prefix}_Precision_Positive": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        f"{prefix}_Recall_Positive": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        f"{prefix}_F1_Positive": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        f"{prefix}_Specificity": specificity,
        f"{prefix}_ROC_AUC": roc_auc,
        f"{prefix}_TN": int(tn),
        f"{prefix}_FP": int(fp),
        f"{prefix}_FN": int(fn),
        f"{prefix}_TP": int(tp),
    }


class AnubisExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = resolve_device(args.device)
        self.dtype = torch.double
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dt_csv = self.output_dir / DT_CSV_NAME
        self.classifier_metrics_csv = (
            self.output_dir / CLASSIFIER_METRICS_CSV_NAME
        )
        self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> None:
        dataset = self.args.dataset.resolve()
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        df = pd.read_excel(dataset, sheet_name=self.args.sheet)
        required = ID_COLUMNS + [LABEL_COLUMN, BANDGAP_COLUMN] + DESCRIPTOR_COLUMNS
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        original_rows = len(df)
        df = df.drop_duplicates(
            subset=["molcat", "metal", "halogen"], keep="first"
        ).reset_index(drop=True)
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="raise").astype(int)
        if not df[LABEL_COLUMN].isin([0, 1]).all():
            raise ValueError("The feasible column must contain only 0 and 1.")
        df[BANDGAP_COLUMN] = pd.to_numeric(df[BANDGAP_COLUMN], errors="coerce")
        df[OBJECTIVE_COLUMN] = (
            df[BANDGAP_COLUMN] - self.args.target_bandgap
        ).abs()
        df.loc[df[LABEL_COLUMN] == 0, OBJECTIVE_COLUMN] = np.nan

        feasible_missing = df.loc[
            df[LABEL_COLUMN] == 1, OBJECTIVE_COLUMN
        ].isna()
        if feasible_missing.any():
            count = int(feasible_missing.sum())
            raise ValueError(f"{count} feasible candidates have no valid band gap.")

        X_df = df[DESCRIPTOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        all_nan_columns = X_df.columns[X_df.isna().all()].tolist()
        if all_nan_columns:
            print(f"Dropping all-NaN descriptors: {all_nan_columns}", flush=True)
            X_df = X_df.drop(columns=all_nan_columns)
        if X_df.shape[1] == 0:
            raise ValueError("No usable descriptor columns remain.")

        imputer = SimpleImputer(strategy="median")
        X_all = imputer.fit_transform(X_df).astype(float)
        x_min = X_all.min(axis=0)
        x_range = np.maximum(X_all.max(axis=0) - x_min, 1e-12)

        self.df = df
        self.feature_columns = X_df.columns.tolist()
        self.X_unit = (X_all - x_min) / x_range
        self.labels = df[LABEL_COLUMN].to_numpy(dtype=int)
        self.f1 = df[OBJECTIVE_COLUMN].to_numpy(dtype=float)
        self.bandgaps = df[BANDGAP_COLUMN].to_numpy(dtype=float)
        self.candidate_ids = df["candidate_id"].to_numpy()
        self.molcats = df["molcat"].astype(str).to_numpy()
        self.metals = df["metal"].astype(str).to_numpy()
        self.halogens = df["halogen"].astype(str).to_numpy()
        self.formulas = (
            df["molcat"].astype(str)
            + "-"
            + df["metal"].astype(str)
            + "-"
            + df["halogen"].astype(str)
        ).to_numpy()

        self.n_candidates = len(df)
        self.feasible_indices = np.where(self.labels == 1)[0]
        self.infeasible_indices = np.where(self.labels == 0)[0]
        self.n_initial = self.args.initial_feasible + self.args.initial_infeasible
        self.global_best_idx = int(
            self.feasible_indices[
                np.argmin(self.f1[self.feasible_indices])
            ]
        )
        self.global_best_f1 = float(self.f1[self.global_best_idx])

        if self.args.initial_feasible > len(self.feasible_indices):
            raise ValueError("Initial feasible count exceeds available candidates.")
        if self.args.initial_infeasible > len(self.infeasible_indices):
            raise ValueError("Initial infeasible count exceeds available candidates.")
        if self.n_initial + self.args.iterations > self.n_candidates:
            raise ValueError(
                f"Requested {self.n_initial + self.args.iterations} evaluations, "
                f"but only {self.n_candidates} unique candidates exist."
            )

        print(f"Dataset: {dataset}", flush=True)
        print(
            f"Rows: {original_rows}; unique candidates: {self.n_candidates}; "
            f"feasible: {len(self.feasible_indices)}; "
            f"infeasible: {len(self.infeasible_indices)}",
            flush=True,
        )
        print(
            f"Features: {len(self.feature_columns)}; device: {self.device}; "
            f"global minimum f1: {self.global_best_f1:.8g}",
            flush=True,
        )

    def make_initial_indices(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        feasible = rng.choice(
            self.feasible_indices,
            size=self.args.initial_feasible,
            replace=False,
        )
        infeasible = rng.choice(
            self.infeasible_indices,
            size=self.args.initial_infeasible,
            replace=False,
        )
        indices = np.concatenate([feasible, infeasible]).astype(int)
        rng.shuffle(indices)
        return indices

    def observed_feasible_indices(self, observed: np.ndarray) -> np.ndarray:
        return np.where(observed & (self.labels == 1) & np.isfinite(self.f1))[0]

    def best_observed_idx(self, observed: np.ndarray) -> int:
        indices = self.observed_feasible_indices(observed)
        if indices.size == 0:
            raise RuntimeError("No feasible objective has been observed.")
        return int(indices[np.argmin(self.f1[indices])])

    def fit_gp(self, train_indices: np.ndarray) -> SingleTaskGP:
        train_indices = np.asarray(train_indices, dtype=int)
        if train_indices.size == 0:
            raise ValueError("The objective GP needs a feasible observation.")
        X_train = torch.as_tensor(
            self.X_unit[train_indices], device=self.device, dtype=self.dtype
        )
        Y_train = torch.as_tensor(
            self.f1[train_indices], device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        model = SingleTaskGP(
            X_train,
            Y_train,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        return model

    def choose_candidate(
        self,
        model: SingleTaskGP,
        best_f1: float,
        search_indices: np.ndarray,
    ) -> tuple[int, float, float, float]:
        search_indices = np.asarray(search_indices, dtype=int)
        acquisition = LogExpectedImprovement(
            model=model,
            best_f=float(best_f1),
            maximize=False,
        )
        choices = torch.as_tensor(
            self.X_unit[search_indices], device=self.device, dtype=self.dtype
        )
        candidate, acquisition_value = optimize_acqf_discrete(
            acq_function=acquisition,
            q=1,
            choices=choices,
            unique=True,
        )
        candidate_np = candidate.squeeze(0).detach().cpu().numpy()
        distances = np.linalg.norm(
            self.X_unit[search_indices] - candidate_np, axis=1
        )
        chosen = int(search_indices[np.argmin(distances)])

        x_chosen = torch.as_tensor(
            self.X_unit[chosen], device=self.device, dtype=self.dtype
        ).reshape(1, -1)
        with torch.no_grad():
            posterior = model.posterior(x_chosen)
            gp_mean = float(posterior.mean.squeeze().detach().cpu())
            gp_variance = max(
                float(posterior.variance.squeeze().detach().cpu()), 0.0
            )
        return (
            chosen,
            gp_mean,
            float(np.sqrt(gp_variance)),
            float(acquisition_value.squeeze().detach().cpu()),
        )

    def candidate_fields(self, index: int) -> dict:
        return {
            "Dataset_Index": int(index),
            "Candidate_ID": safe_scalar(self.candidate_ids[index]),
            "Formula": self.formulas[index],
            "Molcat": self.molcats[index],
            "Metal": self.metals[index],
            "Halogen": self.halogens[index],
            "True_Feasible": int(self.labels[index]),
            "True_Label": "Feasible" if self.labels[index] == 1 else "Infeasible",
            "Bandgap_eV": (
                float(self.bandgaps[index])
                if np.isfinite(self.bandgaps[index])
                else np.nan
            ),
            "Selected_f1": (
                float(self.f1[index]) if np.isfinite(self.f1[index]) else np.nan
            ),
            "Objective_Available": int(np.isfinite(self.f1[index])),
            "Ignored_By_Objective_GP": int(not np.isfinite(self.f1[index])),
        }

    def common_fields(
        self,
        seed: int,
        iteration: int,
        initial_indices: np.ndarray,
        observed: np.ndarray,
        chosen: int,
        gp_mean: float,
        gp_sd: float,
        acquisition_value: float,
    ) -> dict:
        best_idx = self.best_observed_idx(observed)
        global_in_initial = int(self.global_best_idx in initial_indices)
        global_found = int(observed[self.global_best_idx])
        initial_mask = np.zeros(self.n_candidates, dtype=bool)
        initial_mask[initial_indices] = True
        initial_best_idx = self.best_observed_idx(initial_mask)
        return {
            "Seed": int(seed),
            "BO_Iteration": int(iteration),
            "Total_Evaluations": int(self.n_initial + iteration),
            **self.candidate_fields(chosen),
            "GP_Predicted_f1": gp_mean,
            "GP_Predicted_f1_SD": gp_sd,
            "Log_Expected_Improvement": acquisition_value,
            "GP_Updated": int(
                self.labels[chosen] == 1 and np.isfinite(self.f1[chosen])
            ),
            "Best_Feasible_f1": float(self.f1[best_idx]),
            "Best_Feasible_f1_So_Far": float(self.f1[best_idx]),
            "Best_So_Far_Dataset_Index": int(best_idx),
            "Best_So_Far_Candidate_ID": safe_scalar(self.candidate_ids[best_idx]),
            "Best_So_Far_Formula": self.formulas[best_idx],
            "Best_So_Far_Bandgap_eV": float(self.bandgaps[best_idx]),
            "Initial_Best_Feasible_f1": float(self.f1[initial_best_idx]),
            "Global_Best_Dataset_Index": int(self.global_best_idx),
            "Global_Best_f1": self.global_best_f1,
            "Global_Best_In_Initial": global_in_initial,
            "Global_Best_Found_By_This_Iteration": global_found,
            "Candidate_Space_Size": int(self.n_candidates),
            "Initial_Feasible_Count": int(self.args.initial_feasible),
            "Initial_Infeasible_Count": int(self.args.initial_infeasible),
            "Initial_Evaluations": int(self.n_initial),
            "Target_Bandgap_eV": float(self.args.target_bandgap),
        }

    def make_classifier(self, seed: int) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.args.tree_max_depth,
            min_samples_split=self.args.min_samples_split,
            min_samples_leaf=self.args.min_samples_leaf,
            class_weight="balanced",
            random_state=seed,
        )

    def classifier_diagnostic(
        self,
        classifier: DecisionTreeClassifier,
        seed: int,
        refit_iteration: int,
        training_indices: list[int],
        observed: np.ndarray,
    ) -> dict:
        """Evaluate the fitted DT; predictions are diagnostic-only."""
        # Predicting here is read-only. No full-dataset label or prediction is
        # appended to training_indices or used by the acquisition function.
        full_pred = classifier.predict(self.X_unit).astype(int)
        class_position = int(np.where(classifier.classes_ == 1)[0][0])
        full_score = classifier.predict_proba(self.X_unit)[:, class_position]
        unobserved_indices = np.where(~observed)[0]

        row = {
            "Seed": int(seed),
            "Refit_After_BO_Iteration": int(refit_iteration),
            "Total_Evaluations": int(self.n_initial + refit_iteration),
            "Classifier_Training_Points": int(len(training_indices)),
            "Positive_Class": "Feasible",
            "Negative_Class": "Infeasible",
            "Diagnostic_Only": 1,
            "Predictions_Used_For_Training": 0,
        }
        row.update(
            binary_metric_values(
                self.labels, full_pred, full_score, "Full_Dataset"
            )
        )
        if unobserved_indices.size:
            row.update(
                binary_metric_values(
                    self.labels[unobserved_indices],
                    full_pred[unobserved_indices],
                    full_score[unobserved_indices],
                    "Unobserved",
                )
            )
        else:
            row.update(
                binary_metric_values(
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=float),
                    "Unobserved",
                )
            )
        return row

    def run_dt_bo_seed(
        self, seed: int, initial_indices: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        set_all_seeds(seed)
        # Restarting here matches the supplied notebook's BO-stage RNG behavior.
        rng = np.random.default_rng(seed)
        observed = np.zeros(self.n_candidates, dtype=bool)
        observed[initial_indices] = True
        classifier_train = initial_indices.astype(int).tolist()
        pending_classifier_points: list[int] = []

        classifier = self.make_classifier(seed)
        classifier.fit(
            self.X_unit[classifier_train], self.labels[classifier_train]
        )
        records = []
        classifier_metrics = []

        for iteration in range(1, self.args.iterations + 1):
            remaining = np.where(~observed)[0]
            if remaining.size == 0:
                break
            gp_train = self.observed_feasible_indices(observed)
            gp_training_points_before = int(len(gp_train))
            model = self.fit_gp(gp_train)

            predicted_labels = classifier.predict(self.X_unit[remaining])
            predicted_feasible = remaining[predicted_labels == 1]
            if predicted_feasible.size > 0 and rng.random() <= self.args.alpha:
                search_indices = predicted_feasible
                search_mode = "DT-predicted feasible"
            else:
                search_indices = remaining
                search_mode = "Full-space exploration"

            best_before = float(self.f1[self.best_observed_idx(observed)])
            chosen, gp_mean, gp_sd, acq = self.choose_candidate(
                model, best_before, search_indices
            )
            dt_prediction = int(
                classifier.predict(self.X_unit[chosen].reshape(1, -1))[0]
            )
            observed[chosen] = True
            pending_classifier_points.append(chosen)

            row = self.common_fields(
                seed,
                iteration,
                initial_indices,
                observed,
                chosen,
                gp_mean,
                gp_sd,
                acq,
            )
            row.update(
                {
                    "DT_Predicted_Feasible": dt_prediction,
                    "DT_Predicted_Label": (
                        "Feasible" if dt_prediction == 1 else "Infeasible"
                    ),
                    "DT_Prediction_Correct": int(
                        dt_prediction == self.labels[chosen]
                    ),
                    "Search_Mode": search_mode,
                    "Search_Set_Size": int(len(search_indices)),
                    "Predicted_Feasible_Remaining": int(len(predicted_feasible)),
                    "GP_Training_Points_Before_Selection": gp_training_points_before,
                    "GP_Training_Points_After_Selection": (
                        gp_training_points_before
                        + int(self.labels[chosen] == 1 and np.isfinite(self.f1[chosen]))
                    ),
                    "Objective_GP_Action": (
                        "Added feasible objective to GP"
                        if self.labels[chosen] == 1 and np.isfinite(self.f1[chosen])
                        else "Ignored infeasible result; propose again next iteration"
                    ),
                }
            )
            records.append(row)

            if self.args.refit_every and iteration % self.args.refit_every == 0:
                classifier_train = np.unique(
                    classifier_train + pending_classifier_points
                ).astype(int).tolist()
                classifier = self.make_classifier(seed)
                classifier.fit(
                    self.X_unit[classifier_train], self.labels[classifier_train]
                )
                classifier_metrics.append(
                    self.classifier_diagnostic(
                        classifier,
                        seed,
                        iteration,
                        classifier_train,
                        observed,
                    )
                )
                pending_classifier_points = []

            if iteration == 1 or iteration % 25 == 0:
                print(
                    f"  DT-BO seed {seed:02d}, iteration "
                    f"{iteration:03d}/{self.args.iterations}: "
                    f"best f1={row['Best_Feasible_f1']:.8g}",
                    flush=True,
                )

        return pd.DataFrame(records), pd.DataFrame(classifier_metrics)

    @staticmethod
    def checkpoint(frames: list[pd.DataFrame], destination: Path) -> None:
        pd.concat(frames, ignore_index=True).to_csv(destination, index=False)

    def run(self):
        seed_list = list(
            range(self.args.seed_start, self.args.seed_start + self.args.seeds)
        )
        dt_frames: list[pd.DataFrame] = []
        classifier_metric_frames: list[pd.DataFrame] = []
        started = time.time()

        for run_number, seed in enumerate(seed_list, start=1):
            print(
                f"\nSeed {seed} ({run_number}/{len(seed_list)})",
                flush=True,
            )
            initial_indices = self.make_initial_indices(seed)
            dt_seed, classifier_seed = self.run_dt_bo_seed(
                seed, initial_indices
            )
            dt_frames.append(dt_seed)
            classifier_metric_frames.append(classifier_seed)
            self.checkpoint(dt_frames, self.dt_csv)
            self.checkpoint(
                classifier_metric_frames, self.classifier_metrics_csv
            )

            elapsed_hours = (time.time() - started) / 3600.0
            print(
                f"Finished seed {seed}; elapsed {elapsed_hours:.2f} h; "
                "combined CSV checkpoints updated.",
                flush=True,
            )

        dt_all = pd.concat(dt_frames, ignore_index=True)
        classifier_metrics_all = pd.concat(
            classifier_metric_frames, ignore_index=True
        )
        expected_rows = self.args.seeds * self.args.iterations
        if len(dt_all) != expected_rows:
            raise RuntimeError(
                f"Expected {expected_rows} DT-BO rows; got {len(dt_all)}."
            )
        expected_refits = self.args.seeds * (
            self.args.iterations // self.args.refit_every
        )
        if len(classifier_metrics_all) != expected_refits:
            raise RuntimeError(
                f"Expected {expected_refits} classifier-refit metric rows, "
                f"but obtained {len(classifier_metrics_all)}."
            )
        return dt_all, classifier_metrics_all


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = AnubisExperiment(args)
    print(
        f"Running {args.seeds} seeds × {args.iterations} iterations for DT-BO.",
        flush=True,
    )
    experiment.run()
    print(f"\nDT-BO CSV: {experiment.dt_csv}", flush=True)
    print(
        f"DT classifier metrics CSV: {experiment.classifier_metrics_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Completed-seed CSV checkpoints were retained.", file=sys.stderr)
        raise SystemExit(130)
