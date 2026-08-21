import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EVAL_THRESHOLD = 0.70
CLASS_LABELS = (0, 1, 2)


def _build_model(params: dict):
    """Create a supported classifier from a serializable parameter mapping."""
    model_params = dict(params)
    model_type = model_params.pop("model_type", "random_forest")

    if model_type == "random_forest":
        model_params.setdefault("random_state", 42)
        model_params.setdefault("n_jobs", -1)
        return model_type, RandomForestClassifier(**model_params)

    if model_type == "gradient_boosting":
        model_params.setdefault("random_state", 42)
        return model_type, GradientBoostingClassifier(**model_params)

    if model_type == "logistic_regression":
        model_params.setdefault("max_iter", 1000)
        model_params.setdefault("random_state", 42)
        return model_type, make_pipeline(
            StandardScaler(), LogisticRegression(**model_params)
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _label_distribution(y: pd.Series) -> dict[str, float]:
    distribution = y.value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
    return {str(label): float(distribution[label]) for label in CLASS_LABELS}


def _write_report(y_true, predictions, report_path: Path) -> None:
    matrix = confusion_matrix(y_true, predictions, labels=CLASS_LABELS)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=CLASS_LABELS,
        zero_division=0,
    )

    lines = [
        "CONFUSION MATRIX (rows=true, columns=predicted)",
        "labels: 0 1 2",
        *[" ".join(str(value) for value in row) for row in matrix],
        "",
        "PER-CLASS METRICS",
        "class precision recall f1 support",
    ]
    for index, label in enumerate(CLASS_LABELS):
        lines.append(
            f"{label} {precision[index]:.4f} {recall[index]:.4f} "
            f"{f1_per_class[index]:.4f} {int(support[index])}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
    tracking_enabled: bool = True,
) -> float:
    """Train, evaluate, track and persist a Wine Quality classifier."""
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    model_type, model = _build_model(params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_eval)
    accuracy = float(accuracy_score(y_eval, predictions))
    weighted_f1 = float(f1_score(y_eval, predictions, average="weighted"))
    label_distribution = _label_distribution(y_train)

    for label, ratio in label_distribution.items():
        if ratio < 0.10:
            print(f"WARNING: class {label} only represents {ratio:.2%} of training data")

    output_dir = Path("outputs")
    model_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    metrics = {
        "accuracy": accuracy,
        "f1_score": weighted_f1,
        "model_type": model_type,
        "train_size": int(len(df_train)),
        "eval_size": int(len(df_eval)),
        "eval_threshold": EVAL_THRESHOLD,
        "label_distribution": label_distribution,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(y_eval, predictions, output_dir / "report.txt")
    joblib.dump(model, model_dir / "model.pkl")

    print(f"Model: {model_type} | Accuracy: {accuracy:.4f} | F1: {weighted_f1:.4f}")

    if tracking_enabled:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality-task-1")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_params({"model_type": model_type, **params})
            mlflow.log_metrics({"accuracy": accuracy, "f1_score": weighted_f1})
            mlflow.log_artifact(str(output_dir / "report.txt"), "evaluation")
            mlflow.sklearn.log_model(model, "model")

    return accuracy


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as params_file:
        train(yaml.safe_load(params_file))
