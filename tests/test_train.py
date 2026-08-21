import os
import json
import numpy as np
import pandas as pd
import pytest
from src.train import train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path, monkeypatch):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)

    acc = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
        tracking_enabled=False,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path, monkeypatch):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
        tracking_enabled=False,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "label_distribution" in metrics
    assert sum(metrics["label_distribution"].values()) == pytest.approx(1.0)


def test_model_and_report_files_created(tmp_path, monkeypatch):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
        tracking_enabled=False,
    )

    assert os.path.exists("models/model.pkl")
    report = (tmp_path / "outputs" / "report.txt").read_text(encoding="utf-8")
    assert "CONFUSION MATRIX" in report
    assert "precision recall" in report


@pytest.mark.parametrize("model_type", ["gradient_boosting", "logistic_regression"])
def test_bonus_model_types_train(model_type, tmp_path, monkeypatch):
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)

    acc = train(
        {"model_type": model_type, "n_estimators": 10}
        if model_type == "gradient_boosting"
        else {"model_type": model_type},
        data_path=train_path,
        eval_path=eval_path,
        tracking_enabled=False,
    )

    assert 0.0 <= acc <= 1.0


def test_unknown_model_type_is_rejected(tmp_path, monkeypatch):
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Unsupported model_type"):
        train(
            {"model_type": "not-a-model"},
            data_path=train_path,
            eval_path=eval_path,
            tracking_enabled=False,
        )
