
"""
KNN training module for obesity classification.

This script converts the original notebook workflow into a modular, reusable
Python module. It supports:
- dataset loading from a YAML config file
- artifact management (encoders, scalers, models, figures)
- preprocessing with feature engineering and encoding
- baseline KNN training
- GridSearchCV optimization
- Optuna optimization
- feature subset experiments (all features, BMI only, Weight+Height, Weight+BMI)
- diagnostic plots and artifact persistence

Execution with optune and without it:
    python knn_obesity_module.py --config ../config.yaml --run-optuna
    python knn_obesity_module.py --config ../config.yaml --skip-optuna 
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as st
import seaborn as sns
import yaml
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================
# Configuration containers
# =========================

@dataclass(frozen=True)
class FeatureConfig:
    """Column groups used during preprocessing."""

    target: str = "NObeyesdad"
    numerical: Tuple[str, ...] = (
        "Age",
        "Height",
        "Weight",
        "FCVC",
        "NCP",
        "CH2O",
        "FAF",
        "TUE",
    )
    binary: Tuple[str, ...] = (
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "SMOKE",
        "SCC",
    )
    ordinal: Tuple[str, ...] = ("CAEC", "CALC")
    nominal: Tuple[str, ...] = ("MTRANS",)
    binary_map: Dict[str, int] = field(
        default_factory=lambda: {
            "Male": 0,
            "Female": 1,
            "male": 0,
            "female": 1,
            "no": 0,
            "yes": 1,
            "No": 0,
            "Yes": 1,
        }
    )
    ordinal_map: Dict[str, int] = field(
        default_factory=lambda: {
            "no": 0,
            "No": 0,
            "Sometimes": 1,
            "Frequently": 2,
            "Always": 3,
        }
    )


@dataclass(frozen=True)
class SearchConfig:
    """Search spaces and CV settings for model tuning."""

    test_size: float = 0.20
    random_state: int = 42
    cv_folds_grid: int = 5
    cv_folds_optuna: int = 10
    optuna_trials: int = 45
    confidence_level: float = 0.95
    baseline_neighbors: int = 5
    grid_n_neighbors: Tuple[int, ...] = tuple(range(3, 16))
    optuna_n_neighbors_min: int = 2
    optuna_n_neighbors_max: int = 25


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths."""

    project_dir: Path
    data_file: Path
    encoders_dir: Path
    scalers_dir: Path
    models_dir: Path
    figures_dir: Path


# =========================
# Path and file utilities
# =========================

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_paths(config_path: Path, data_file: Optional[Path] = None) -> ProjectPaths:
    """
    Resolve project paths relative to the configuration file.

    Assumes the config file lives inside the project and the artifact folders
    are siblings of the notebook/scripts directories.
    """
    config_path = config_path.resolve()
    project_dir = config_path.parent

    if data_file is None:
        config = load_yaml_config(config_path)
        raw_path = config["input_data"]["file_1"]
        data_file = (project_dir / raw_path).resolve()

    encoders_dir = project_dir / "encoders"
    scalers_dir = project_dir / "scalers"
    models_dir = project_dir / "models"
    figures_dir = project_dir / "figures"

    for folder in (encoders_dir, scalers_dir, models_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_dir=project_dir,
        data_file=Path(data_file),
        encoders_dir=encoders_dir,
        scalers_dir=scalers_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
    )


def save_joblib_artifact(obj: Any, output_path: Path) -> Path:
    """Persist a Python object with joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output_path)
    return output_path


def save_json_artifact(payload: Dict[str, Any], output_path: Path) -> Path:
    """Save a dictionary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_path


def save_current_figure(output_path: Path, dpi: int = 300) -> Path:
    """Save the current matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


# =========================
# Data loading and prep
# =========================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(csv_path)


def basic_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a lightweight dataset summary for reporting or logging."""
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    return {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "categorical_columns": categorical_cols,
        "target_unique_values": df["NObeyesdad"].nunique() if "NObeyesdad" in df.columns else None,
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated rows from the dataset."""
    return df.drop_duplicates().reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and target."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# =========================
# Preprocessing
# =========================

def add_bmi_feature(X: pd.DataFrame) -> pd.DataFrame:
    """Create the BMI feature from Weight and Height."""
    X = X.copy()
    X["BMI"] = X["Weight"] / (X["Height"] ** 2)
    return X


def encode_binary_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    binary_cols: Iterable[str],
    binary_map: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode binary columns using a fixed mapping."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in binary_cols:
        X_train[col] = X_train[col].map(binary_map)
        X_test[col] = X_test[col].map(binary_map)

    return X_train, X_test


def encode_ordinal_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    ordinal_cols: Iterable[str],
    ordinal_map: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode ordinal columns preserving order."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in ordinal_cols:
        X_train[col] = X_train[col].map(ordinal_map)
        X_test[col] = X_test[col].map(ordinal_map)

    return X_train, X_test


def one_hot_encode_nominal_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    nominal_cols: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode nominal features and align test columns."""
    X_train = pd.get_dummies(X_train, columns=list(nominal_cols), drop_first=True)
    X_test = pd.get_dummies(X_test, columns=list(nominal_cols), drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def validate_no_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Raise an error if preprocessing created missing values."""
    train_missing = X_train.isna().sum()
    test_missing = X_test.isna().sum()

    if train_missing.any():
        cols = train_missing[train_missing > 0].to_dict()
        raise ValueError(f"Missing values found in training features after preprocessing: {cols}")
    if test_missing.any():
        cols = test_missing[test_missing > 0].to_dict()
        raise ValueError(f"Missing values found in test features after preprocessing: {cols}")


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_config: FeatureConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full feature preprocessing workflow."""
    X_train = add_bmi_feature(X_train)
    X_test = add_bmi_feature(X_test)

    X_train, X_test = encode_binary_columns(
        X_train, X_test, feature_config.binary, feature_config.binary_map
    )
    X_train, X_test = encode_ordinal_columns(
        X_train, X_test, feature_config.ordinal, feature_config.ordinal_map
    )
    X_train, X_test = one_hot_encode_nominal_columns(
        X_train, X_test, feature_config.nominal
    )

    validate_no_missing_values(X_train, X_test)
    return X_train, X_test


def encode_target(
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, Dict[str, int]]:
    """Fit a LabelEncoder on the training target and encode both sets."""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    class_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    return y_train_encoded, y_test_encoded, label_encoder, class_mapping


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on training data and transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def create_feature_subset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    subset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create reusable feature subsets used in the notebook experiments."""
    subset_name = subset_name.lower()

    if subset_name == "all":
        return X_train.copy(), X_test.copy()
    if subset_name == "bmi_only":
        return X_train.drop(columns=["Weight", "Height"]), X_test.drop(columns=["Weight", "Height"])
    if subset_name == "weight_height":
        return X_train.drop(columns=["BMI"]), X_test.drop(columns=["BMI"])
    if subset_name == "weight_bmi":
        return X_train.drop(columns=["Height"]), X_test.drop(columns=["Height"])

    raise ValueError(
        "Invalid subset_name. Expected one of: 'all', 'bmi_only', 'weight_height', 'weight_bmi'."
    )


# =========================
# Modeling
# =========================

def build_baseline_knn(n_neighbors: int) -> KNeighborsClassifier:
    """Create a baseline KNN classifier."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def fit_model(model: ClassifierMixin, X_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    """Fit any sklearn-compatible classifier."""
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Iterable[str],
) -> Dict[str, Any]:
    """Evaluate a classifier and return the main outputs."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=list(class_names)
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_config: SearchConfig,
) -> GridSearchCV:
    """Tune KNN with GridSearchCV."""
    param_grid = {
        "n_neighbors": list(search_config.grid_n_neighbors),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    grid = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=search_config.cv_folds_grid,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def make_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    confidence_level: float,
    folds: int,
):
    """Create the Optuna objective function with enclosed training data."""

    def objective(trial: optuna.Trial) -> float:
        n_neighbors = trial.suggest_int("n_neighbors", 2, 25)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 3)

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
        )

        scores = cross_val_score(
            knn,
            X_train,
            y_train,
            cv=folds,
            scoring="accuracy",
        )

        mean_score = float(np.mean(scores))
        sem = np.std(scores, ddof=1) / np.sqrt(folds)
        tc = st.t.ppf(1 - ((1 - confidence_level) / 2), df=folds - 1)
        lower_bound = mean_score - (tc * sem)
        upper_bound = mean_score + (tc * sem)

        trial.set_user_attr(
            "CV_score_summary",
            [round(lower_bound, 4), round(mean_score, 4), round(upper_bound, 4)],
        )

        return mean_score

    return objective


def run_optuna_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_config: SearchConfig,
) -> optuna.study.Study:
    """Tune KNN with Optuna."""
    study = optuna.create_study(direction="maximize")
    objective = make_optuna_objective(
        X_train=X_train,
        y_train=y_train,
        confidence_level=search_config.confidence_level,
        folds=search_config.cv_folds_optuna,
    )
    study.optimize(objective, n_trials=search_config.optuna_trials)
    return study


def train_optuna_best_model(study: optuna.study.Study, X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    """Train the final KNN model from the best Optuna parameters."""
    model = KNeighborsClassifier(
        n_neighbors=study.best_params["n_neighbors"],
        weights=study.best_params["weights"],
        p=study.best_params["p"],
    )
    model.fit(X_train, y_train)
    return model


# =========================
# Diagnostics and plotting
# =========================

def compute_vif_table(X_train_scaled: np.ndarray, feature_names: Iterable[str]) -> pd.DataFrame:
    """Compute VIF for a scaled feature matrix."""
    X_vif = pd.DataFrame(X_train_scaled, columns=list(feature_names))
    vif_data = pd.DataFrame({"feature": X_vif.columns})
    vif_data["VIF"] = [
        variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def create_correlation_heatmap(
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> pd.DataFrame:
    """Create and optionally save an absolute-correlation heatmap."""
    df_corr = X_train.copy()
    df_corr["target"] = y_train_encoded

    corr = np.abs(df_corr.corr())
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        annot=False,
    )
    plt.title("Feature Correlation Heatmap (Absolute Values)")

    if output_path is not None:
        save_current_figure(output_path)

    return corr


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Iterable[str],
    output_path: Optional[Path] = None,
    normalized: bool = False,
    title: Optional[str] = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    if normalized:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=list(class_names),
            normalize="true",
        )
    else:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
        disp.plot(xticks_rotation=45)

    if title:
        plt.title(title)

    if output_path is not None:
        save_current_figure(output_path)


def compute_permutation_importance_table(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Iterable[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance for a fitted classifier."""
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy",
    )
    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "importance_mean": perm.importances_mean,
            }
        )
        .sort_values(by="importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n: int = 15,
) -> None:
    """Plot the top permutation importances."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(top_n), x="importance_mean", y="feature")
    plt.title("Top Feature Importance (Permutation)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    if output_path is not None:
        save_current_figure(output_path)


# =========================
# Reporting helpers
# =========================

def build_comparison_table(
    grid: GridSearchCV,
    grid_test_accuracy: float,
    study: optuna.study.Study,
    optuna_test_accuracy: float,
) -> pd.DataFrame:
    """Create a side-by-side comparison table for GridSearch and Optuna."""
    return pd.DataFrame(
        [
            {
                "Method": "GridSearchCV",
                "n_neighbors": grid.best_params_["n_neighbors"],
                "weights": grid.best_params_["weights"],
                "metric_or_p": grid.best_params_.get("metric", "N/A"),
                "Best CV Score": round(grid.best_score_, 4),
                "Test Accuracy": round(grid_test_accuracy, 4),
            },
            {
                "Method": "Optuna",
                "n_neighbors": study.best_params["n_neighbors"],
                "weights": study.best_params["weights"],
                "metric_or_p": study.best_params["p"],
                "Best CV Score": round(study.best_value, 4),
                "Test Accuracy": round(optuna_test_accuracy, 4),
            },
        ]
    )


# =========================
# Orchestration
# =========================

def run_subset_experiment(
    subset_name: str,
    X_train_processed: pd.DataFrame,
    X_test_processed: pd.DataFrame,
    y_train_encoded: np.ndarray,
    y_test_encoded: np.ndarray,
    class_names: Iterable[str],
    paths: ProjectPaths,
    search_config: SearchConfig,
) -> Dict[str, Any]:
    """Run baseline + GridSearch on a specific feature subset."""
    X_train_subset, X_test_subset = create_feature_subset(
        X_train_processed, X_test_processed, subset_name
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_subset, X_test_subset)

    scaler_name = f"standard_scaler_{subset_name}.pkl"
    save_joblib_artifact(scaler, paths.scalers_dir / scaler_name)

    baseline_model = fit_model(
        build_baseline_knn(search_config.baseline_neighbors),
        X_train_scaled,
        y_train_encoded,
    )
    baseline_results = evaluate_classifier(
        baseline_model, X_test_scaled, y_test_encoded, class_names
    )
    baseline_model_name = f"knn_baseline_{subset_name}.pkl"
    save_joblib_artifact(baseline_model, paths.models_dir / baseline_model_name)

    grid = run_grid_search(X_train_scaled, y_train_encoded, search_config)
    best_grid_model = grid.best_estimator_
    grid_results = evaluate_classifier(
        best_grid_model, X_test_scaled, y_test_encoded, class_names
    )

    save_joblib_artifact(best_grid_model, paths.models_dir / f"knn_gridsearch_{subset_name}.pkl")
    save_joblib_artifact(grid, paths.models_dir / f"gridsearch_{subset_name}.pkl")

    return {
        "subset_name": subset_name,
        "X_train": X_train_subset,
        "X_test": X_test_subset,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "baseline_model": baseline_model,
        "baseline_results": baseline_results,
        "grid": grid,
        "best_grid_model": best_grid_model,
        "grid_results": grid_results,
    }


def run_full_workflow(
    config_path: Path,
    feature_config: Optional[FeatureConfig] = None,
    search_config: Optional[SearchConfig] = None,
    run_optuna: bool = True,
) -> Dict[str, Any]:
    """
    Run the full modeling workflow using all features and subset experiments.

    Returns a dictionary of the key artifacts and results so the module can be
    imported and reused programmatically.
    """
    feature_config = feature_config or FeatureConfig()
    search_config = search_config or SearchConfig()
    paths = resolve_project_paths(config_path)

    # Load and prepare data.
    df = load_dataset(paths.data_file)
    df = remove_duplicates(df)
    X, y = split_features_target(df, feature_config.target)
    X_train, X_test, y_train, y_test = split_train_test(
        X=X,
        y=y,
        test_size=search_config.test_size,
        random_state=search_config.random_state,
    )
    X_train_processed, X_test_processed = preprocess_features(
        X_train, X_test, feature_config
    )

    # Encode target and save encoder artifacts.
    y_train_encoded, y_test_encoded, label_encoder, class_mapping = encode_target(
        y_train, y_test
    )
    save_joblib_artifact(label_encoder, paths.encoders_dir / "label_encoder.pkl")
    save_json_artifact(
        {str(k): int(v) for k, v in class_mapping.items()},
        paths.encoders_dir / "class_mapping.json",
    )

    # Run experiments for all feature sets used in the notebook.
    subset_results = {}
    for subset_name in ("all", "bmi_only", "weight_height", "weight_bmi"):
        subset_results[subset_name] = run_subset_experiment(
            subset_name=subset_name,
            X_train_processed=X_train_processed,
            X_test_processed=X_test_processed,
            y_train_encoded=y_train_encoded,
            y_test_encoded=y_test_encoded,
            class_names=label_encoder.classes_,
            paths=paths,
            search_config=search_config,
        )

    # Core all-feature artifacts.
    all_results = subset_results["all"]
    create_confusion_matrix_plot(
        y_true=y_test_encoded,
        y_pred=all_results["baseline_results"]["y_pred"],
        class_names=label_encoder.classes_,
        output_path=paths.figures_dir / "confusion_matrix_baseline_all_features.png",
        title="Confusion Matrix - Baseline KNN (All Features)",
    )
    create_confusion_matrix_plot(
        y_true=y_test_encoded,
        y_pred=all_results["grid_results"]["y_pred"],
        class_names=label_encoder.classes_,
        output_path=paths.figures_dir / "confusion_matrix_gridsearch_all_features.png",
        title="Confusion Matrix - Best GridSearch KNN (All Features)",
    )

    corr = create_correlation_heatmap(
        X_train_processed,
        y_train_encoded,
        output_path=paths.figures_dir / "feature_correlation_heatmap.png",
    )
    vif_table = compute_vif_table(all_results["X_train_scaled"], X_train_processed.columns)
    feature_importance = compute_permutation_importance_table(
        model=all_results["best_grid_model"],
        X_test=all_results["X_test_scaled"],
        y_test=y_test_encoded,
        feature_names=X_train_processed.columns,
    )
    plot_feature_importance(
        feature_importance,
        output_path=paths.figures_dir / "feature_importance_barplot.png",
    )

    artifacts: Dict[str, Any] = {
        "paths": paths,
        "label_encoder": label_encoder,
        "class_mapping": class_mapping,
        "subset_results": subset_results,
        "correlation_matrix": corr,
        "vif_table": vif_table,
        "feature_importance": feature_importance,
    }

    if run_optuna:
        study = run_optuna_search(
            all_results["X_train_scaled"],
            y_train_encoded,
            search_config,
        )
        best_optuna_model = train_optuna_best_model(
            study,
            all_results["X_train_scaled"],
            y_train_encoded,
        )
        optuna_results = evaluate_classifier(
            best_optuna_model,
            all_results["X_test_scaled"],
            y_test_encoded,
            label_encoder.classes_,
        )

        save_joblib_artifact(best_optuna_model, paths.models_dir / "knn_optuna_all_features.pkl")
        save_joblib_artifact(study, paths.models_dir / "optuna_study_all_features.pkl")

        create_confusion_matrix_plot(
            y_true=y_test_encoded,
            y_pred=optuna_results["y_pred"],
            class_names=label_encoder.classes_,
            output_path=paths.figures_dir / "confusion_matrix_optuna_all_features.png",
            title="Confusion Matrix - Best Optuna KNN (All Features)",
        )

        comparison_df = build_comparison_table(
            grid=all_results["grid"],
            grid_test_accuracy=all_results["grid_results"]["accuracy"],
            study=study,
            optuna_test_accuracy=optuna_results["accuracy"],
        )
        comparison_df.to_csv(
            paths.models_dir / "gridsearch_vs_optuna_comparison.csv",
            index=False,
        )

        artifacts["optuna_study"] = study
        artifacts["optuna_model"] = best_optuna_model
        artifacts["optuna_results"] = optuna_results
        artifacts["comparison_df"] = comparison_df

    return artifacts


# =========================
# Command-line interface
# =========================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a reusable KNN obesity classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file containing the input CSV path.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip the Optuna tuning stage.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the script from the command line."""
    args = parse_args()

    start_time = time.time()
    artifacts = run_full_workflow(
        config_path=args.config,
        run_optuna=not args.skip_optuna,
    )
    elapsed = time.time() - start_time

    all_results = artifacts["subset_results"]["all"]["grid_results"]

    print("\nWorkflow completed successfully.")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Best all-features GridSearch accuracy: {all_results['accuracy']:.4f}")

    if "comparison_df" in artifacts:
        print("\nGridSearch vs Optuna comparison:")
        print(artifacts["comparison_df"])


if __name__ == "__main__":
    main()
