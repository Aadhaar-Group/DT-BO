#!/usr/bin/env python3
"""DT-BO on the high-entropy alloy (HEA) dataset: maximize bulk modulus among single solid-solution (SS) alloys.

Feasible = Phase "SS" (24 of 601 alloys). The GP is trained on observed SS alloys only; a
decision tree (max depth 6, min samples split 5, min samples leaf 2, balanced class weights)
is refitted every 5 iterations on the observed labels; with probability alpha = 0.9 the search
is restricted to the tree's predicted-SS candidates, otherwise all remaining candidates.

Usage:
    python hea_dt_bo.py --dataset data/dataset.xlsx --output-dir results/dt_bo

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


ALLOY_COLUMN = "Alloy"
PHASE_COLUMN = "Phase"
LABEL_COLUMN = "Phase_SS"
OBJECTIVE_COLUMN = "BulkModulus"

FEATURE_COLUMNS = [
    "a",
    "delta",
    "Tm",
    "D_Tm",
    "Hmix",
    "σHmix",
    "Sid",
    "Elec_nega",
    "D_elec_nega",
    "VEC",
    "d_VEC",
]

DT_CSV_NAME = "DT_BO_All_Seeds_Iterations.csv"
CLASSIFIER_METRICS_CSV_NAME = "DT_Classifier_Refit_Metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DT-BO on the HEA (Nature alloy) dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset.xlsx"),
        help="Original or processed .xlsx/.csv dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name. The first sheet is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dt_bo"),
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--initial-ss", type=int, default=5)
    parser.add_argument("--initial-non-ss", type=int, default=5)
    parser.add_argument("--tree-max-depth", type=int, default=6)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--dt-refit-every",
        type=int,
        default=5,
        help="Refit DT after this many BO measurements (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.90,
        help="Probability DT-BO searches only predicted-SS candidates.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device (default: auto).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("seeds", "iterations", "initial_ss", "initial_non_ss"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start cannot be negative.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.dt_refit_every <= 0:
        raise ValueError("--dt-refit-every must be positive.")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    return torch.device(requested)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    prefix: str,
) -> dict:
    """Return binary classification metrics without modifying the DT."""
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


class NatureBulkExperiment:
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

    def _read_dataset(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            sheet = self.args.sheet if self.args.sheet is not None else 0
            return pd.read_excel(path, sheet_name=sheet)
        if suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError("Dataset must be an .xlsx, .xls, or .csv file.")

    def _load_and_prepare_data(self) -> None:
        dataset = self.args.dataset.resolve()
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset}")
        df = self._read_dataset(dataset)
        original_rows = len(df)

        if LABEL_COLUMN not in df.columns:
            if PHASE_COLUMN not in df.columns:
                raise ValueError(
                    f"Dataset needs either {LABEL_COLUMN!r} or {PHASE_COLUMN!r}."
                )
            df[LABEL_COLUMN] = (
                df[PHASE_COLUMN]
                .astype(str)
                .str.strip()
                .str.upper()
                .eq("SS")
                .astype(int)
            )

        required = [ALLOY_COLUMN, LABEL_COLUMN, OBJECTIVE_COLUMN] + FEATURE_COLUMNS
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for column in FEATURE_COLUMNS + [LABEL_COLUMN, OBJECTIVE_COLUMN]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df[FEATURE_COLUMNS + [OBJECTIVE_COLUMN]] = df[
            FEATURE_COLUMNS + [OBJECTIVE_COLUMN]
        ].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[LABEL_COLUMN, OBJECTIVE_COLUMN]).reset_index(drop=True)
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
        if not df[LABEL_COLUMN].isin([0, 1]).all():
            raise ValueError("Phase_SS must contain only 0 and 1.")

        X_df = df[FEATURE_COLUMNS].copy()
        all_nan_columns = X_df.columns[X_df.isna().all()].tolist()
        if all_nan_columns:
            print(f"Dropping all-NaN features: {all_nan_columns}", flush=True)
            X_df = X_df.drop(columns=all_nan_columns)
        if X_df.shape[1] == 0:
            raise ValueError("No usable feature columns remain.")

        imputer = SimpleImputer(strategy="median")
        X_all = imputer.fit_transform(X_df).astype(float)
        x_min = X_all.min(axis=0)
        x_range = np.maximum(X_all.max(axis=0) - x_min, 1e-12)

        self.df = df
        self.features = X_df.columns.tolist()
        self.X_tree = X_all
        self.X_unit = (X_all - x_min) / x_range
        self.labels = df[LABEL_COLUMN].to_numpy(dtype=int)
        self.objective = df[OBJECTIVE_COLUMN].to_numpy(dtype=float)
        self.alloys = df[ALLOY_COLUMN].astype(str).to_numpy()
        self.original_phases = (
            df[PHASE_COLUMN].astype(str).to_numpy()
            if PHASE_COLUMN in df.columns
            else np.where(self.labels == 1, "SS", "Non-SS")
        )
        self.n_candidates = len(df)
        self.ss_indices = np.where(self.labels == 1)[0]
        self.non_ss_indices = np.where(self.labels == 0)[0]
        self.n_initial = self.args.initial_ss + self.args.initial_non_ss

        self.global_best_ss_idx = int(
            self.ss_indices[np.argmax(self.objective[self.ss_indices])]
        )
        self.global_best_non_ss_idx = int(
            self.non_ss_indices[np.argmax(self.objective[self.non_ss_indices])]
        )
        self.global_best_idx = int(np.argmax(self.objective))

        if self.args.initial_ss > len(self.ss_indices):
            raise ValueError("Initial SS count exceeds available SS candidates.")
        if self.args.initial_non_ss > len(self.non_ss_indices):
            raise ValueError("Initial non-SS count exceeds available candidates.")
        if self.n_initial + self.args.iterations > self.n_candidates:
            raise ValueError(
                f"Requested {self.n_initial + self.args.iterations} evaluations, "
                f"but only {self.n_candidates} valid candidates exist."
            )

        print(f"Dataset: {dataset}", flush=True)
        print(
            f"Original rows: {original_rows}; valid candidates: {self.n_candidates}; "
            f"SS: {len(self.ss_indices)}; non-SS: {len(self.non_ss_indices)}",
            flush=True,
        )
        print(
            f"Features: {len(self.features)}; device: {self.device}; "
            f"best SS BulkModulus: {self.objective[self.global_best_ss_idx]:.8g}",
            flush=True,
        )

    def make_initial_indices(self, seed: int) -> tuple[np.ndarray, np.random.Generator]:
        # Returning the advanced RNG preserves the notebook's DT exploration draws.
        rng = np.random.default_rng(seed)
        initial_ss = rng.choice(
            self.ss_indices, size=self.args.initial_ss, replace=False
        )
        initial_non_ss = rng.choice(
            self.non_ss_indices, size=self.args.initial_non_ss, replace=False
        )
        indices = np.concatenate([initial_ss, initial_non_ss]).astype(int)
        rng.shuffle(indices)
        return indices, rng

    def fit_gp(self, train_indices: np.ndarray) -> SingleTaskGP:
        train_indices = np.asarray(train_indices, dtype=int)
        if train_indices.size == 0:
            raise ValueError("The GP has no training observations.")
        X_train = torch.as_tensor(
            self.X_unit[train_indices], dtype=self.dtype, device=self.device
        )
        Y_train = torch.as_tensor(
            self.objective[train_indices], dtype=self.dtype, device=self.device
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
        best_value: float,
        candidate_indices: np.ndarray,
    ) -> tuple[int, float, float, float]:
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        acquisition = LogExpectedImprovement(model=model, best_f=float(best_value))
        choices = torch.as_tensor(
            self.X_unit[candidate_indices], dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            acquisition_values = acquisition(choices.unsqueeze(1)).reshape(-1)
        chosen_local = int(torch.argmax(acquisition_values).item())
        chosen = int(candidate_indices[chosen_local])
        chosen_x = choices[chosen_local].reshape(1, -1)
        with torch.no_grad():
            posterior = model.posterior(chosen_x)
            predicted_mean = float(posterior.mean.squeeze().detach().cpu())
            predicted_variance = max(
                float(posterior.variance.squeeze().detach().cpu()), 0.0
            )
        return (
            chosen,
            predicted_mean,
            float(np.sqrt(predicted_variance)),
            float(acquisition_values[chosen_local].detach().cpu()),
        )

    def best_index(self, observed: np.ndarray, label: int | None = None) -> int:
        mask = observed.copy()
        if label is not None:
            mask &= self.labels == label
        indices = np.where(mask)[0]
        if indices.size == 0:
            raise RuntimeError("No observed candidate exists for the requested group.")
        return int(indices[np.argmax(self.objective[indices])])

    def candidate_fields(self, index: int) -> dict:
        return {
            "Dataset_Index": int(index),
            "Candidate_ID": self.alloys[index],
            "Alloy": self.alloys[index],
            "Original_Phase": self.original_phases[index],
            "True_Feasible": int(self.labels[index]),
            "True_Phase_SS": int(self.labels[index]),
            "True_Label": "SS" if self.labels[index] == 1 else "Non-SS",
            "Selected_Objective": float(self.objective[index]),
            "BulkModulus": float(self.objective[index]),
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
        best_ss = self.best_index(observed, 1)
        best_non_ss = self.best_index(observed, 0)
        best_overall = self.best_index(observed)
        initial_mask = np.zeros(self.n_candidates, dtype=bool)
        initial_mask[initial_indices] = True
        initial_best_ss = self.best_index(initial_mask, 1)
        initial_best_non_ss = self.best_index(initial_mask, 0)
        initial_best_overall = self.best_index(initial_mask)
        return {
            "Seed": int(seed),
            "BO_Iteration": int(iteration),
            "Iteration": int(iteration),
            "Total_Evaluations": int(self.n_initial + iteration),
            **self.candidate_fields(chosen),
            "GP_Predicted_Objective": gp_mean,
            "GP_Predicted_BulkModulus": gp_mean,
            "GP_Predicted_Objective_SD": gp_sd,
            "Log_Expected_Improvement": acquisition_value,
            "Best_Feasible_Objective": float(self.objective[best_ss]),
            "Best_Feasible_Objective_So_Far": float(self.objective[best_ss]),
            "Best_SS_BulkModulus": float(self.objective[best_ss]),
            "Best_SS_Dataset_Index": int(best_ss),
            "Best_SS_Alloy": self.alloys[best_ss],
            "Best_NonSS_BulkModulus": float(self.objective[best_non_ss]),
            "Best_NonSS_Dataset_Index": int(best_non_ss),
            "Best_NonSS_Alloy": self.alloys[best_non_ss],
            "Best_Overall_BulkModulus": float(self.objective[best_overall]),
            "Best_Overall_Dataset_Index": int(best_overall),
            "Best_Overall_Alloy": self.alloys[best_overall],
            "Initial_Best_Feasible_Objective": float(
                self.objective[initial_best_ss]
            ),
            "Initial_Best_SS_BulkModulus": float(
                self.objective[initial_best_ss]
            ),
            "Initial_Best_NonSS_BulkModulus": float(
                self.objective[initial_best_non_ss]
            ),
            "Initial_Best_Overall_BulkModulus": float(
                self.objective[initial_best_overall]
            ),
            "Initial_Dataset_Indices": ";".join(
                str(int(index)) for index in initial_indices
            ),
            "Global_Best_Dataset_Index": int(self.global_best_ss_idx),
            "Global_Best_Objective": float(
                self.objective[self.global_best_ss_idx]
            ),
            "Global_Best_In_Initial": int(
                self.global_best_ss_idx in initial_indices
            ),
            "Global_Best_Found_By_This_Iteration": int(
                observed[self.global_best_ss_idx]
            ),
            "Global_Best_NonSS_Objective": float(
                self.objective[self.global_best_non_ss_idx]
            ),
            "Global_Best_Overall_Objective": float(
                self.objective[self.global_best_idx]
            ),
            "Candidate_Space_Size": int(self.n_candidates),
            "Initial_Feasible_Count": int(self.args.initial_ss),
            "Initial_Infeasible_Count": int(self.args.initial_non_ss),
            "Initial_Evaluations": int(self.n_initial),
            "Optimization_Direction": "maximize",
            "Objective_Name": OBJECTIVE_COLUMN,
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
        training_indices: np.ndarray,
        observed: np.ndarray,
    ) -> dict:
        """Evaluate the fitted DT; predictions are diagnostic-only."""
        # This uses the full labels only to score a frozen fitted tree. Neither
        # predictions nor unobserved labels are added to DT training or BO.
        full_pred = classifier.predict(self.X_tree).astype(int)
        class_position = int(np.where(classifier.classes_ == 1)[0][0])
        full_score = classifier.predict_proba(self.X_tree)[:, class_position]
        unobserved_indices = np.where(~observed)[0]

        row = {
            "Seed": int(seed),
            "Refit_After_BO_Iteration": int(refit_iteration),
            "Total_Evaluations": int(self.n_initial + refit_iteration),
            "Classifier_Training_Points": int(len(training_indices)),
            "Positive_Class": "SS",
            "Negative_Class": "Non-SS",
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
        self,
        seed: int,
        initial_indices: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        set_all_seeds(seed)
        observed = np.zeros(self.n_candidates, dtype=bool)
        observed[initial_indices] = True
        classifier = self.make_classifier(seed)
        classifier.fit(self.X_tree[initial_indices], self.labels[initial_indices])
        records = []
        classifier_metrics = []

        for iteration in range(1, self.args.iterations + 1):
            remaining = np.where(~observed)[0]
            if remaining.size == 0:
                break

            observed_indices = np.where(observed)[0]
            observed_ss = observed_indices[self.labels[observed_indices] == 1]
            model = self.fit_gp(observed_ss)
            predicted_remaining = classifier.predict(self.X_tree[remaining])
            predicted_ss = remaining[predicted_remaining == 1]

            if predicted_ss.size > 0 and rng.random() < self.args.alpha:
                search_indices = predicted_ss
                search_mode = "DT-predicted SS"
            else:
                search_indices = remaining
                search_mode = "Global exploration"

            best_before = float(np.max(self.objective[observed_ss]))
            chosen, gp_mean, gp_sd, acq = self.choose_candidate(
                model, best_before, search_indices
            )
            dt_prediction = int(
                classifier.predict(self.X_tree[chosen].reshape(1, -1))[0]
            )
            observed[chosen] = True

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
                    "DT_Predicted_Phase_SS": dt_prediction,
                    "DT_Predicted_Label": "SS" if dt_prediction == 1 else "Non-SS",
                    "DT_Prediction_Correct": int(
                        dt_prediction == self.labels[chosen]
                    ),
                    "GP_Updated": int(self.labels[chosen] == 1),
                    "Search_Mode": search_mode,
                    "Search_Set_Size": int(len(search_indices)),
                    "Predicted_Feasible_Remaining": int(len(predicted_ss)),
                    "DT_Refit_Every": int(self.args.dt_refit_every),
                    "DT_Refit_Performed_After_Selection": int(
                        iteration % self.args.dt_refit_every == 0
                    ),
                }
            )
            records.append(row)

            # Refit after iterations 5, 10, 15, ... using observed labels only.
            # Full-dataset predictions below are diagnostic and never feed back.
            if iteration % self.args.dt_refit_every == 0:
                training_indices = np.where(observed)[0]
                classifier = self.make_classifier(seed)
                classifier.fit(
                    self.X_tree[training_indices], self.labels[training_indices]
                )
                classifier_metrics.append(
                    self.classifier_diagnostic(
                        classifier,
                        seed,
                        iteration,
                        training_indices,
                        observed,
                    )
                )

            if iteration == 1 or iteration % 25 == 0:
                print(
                    f"  DT-BO seed {seed:02d}, iteration "
                    f"{iteration:03d}/{self.args.iterations}: "
                    f"best SS={row['Best_SS_BulkModulus']:.8g}",
                    flush=True,
                )

        return pd.DataFrame(records), pd.DataFrame(classifier_metrics)

    @staticmethod
    def checkpoint(frames: list[pd.DataFrame], destination: Path) -> None:
        pd.concat(frames, ignore_index=True).to_csv(destination, index=False)

    def run(self):
        seeds = range(self.args.seed_start, self.args.seed_start + self.args.seeds)
        dt_frames: list[pd.DataFrame] = []
        classifier_metric_frames: list[pd.DataFrame] = []
        started = time.time()

        for run_number, seed in enumerate(seeds, start=1):
            print(f"\nSeed {seed} ({run_number}/{self.args.seeds})", flush=True)
            initial_indices, dt_rng = self.make_initial_indices(seed)
            dt_seed, classifier_seed = self.run_dt_bo_seed(
                seed, initial_indices, dt_rng
            )
            dt_frames.append(dt_seed)
            classifier_metric_frames.append(classifier_seed)
            self.checkpoint(dt_frames, self.dt_csv)
            self.checkpoint(
                classifier_metric_frames, self.classifier_metrics_csv
            )

            elapsed = (time.time() - started) / 3600.0
            print(
                f"Finished seed {seed}; elapsed {elapsed:.2f} h; "
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
            self.args.iterations // self.args.dt_refit_every
        )
        if len(classifier_metrics_all) != expected_refits:
            raise RuntimeError(
                f"Expected {expected_refits} classifier-refit metric rows; "
                f"got {len(classifier_metrics_all)}."
            )
        return dt_all, classifier_metrics_all


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = NatureBulkExperiment(args)
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
        print(
            "Interrupted. Completed-seed CSV checkpoints were retained.",
            file=sys.stderr,
        )
        raise SystemExit(130)
