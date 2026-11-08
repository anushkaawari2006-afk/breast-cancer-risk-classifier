"""
train.py - Train, evaluate and save the Breast Cancer Risk Classifier.

Workflow
  1. Load the Breast Cancer Wisconsin (Diagnostic) dataset (569 real cases).
  2. Hold out 20% of the data as a final test set that is used only once.
  3. Compare four models with 5-fold cross-validation on the training data.
  4. Pick the best model, then choose a decision threshold that catches at
     least 98% of malignant cases (using cross-validated predictions, so the
     test set is never used to tune anything).
  5. Evaluate once on the held-out test set, with bootstrap confidence intervals.
  6. Save the model, metrics and charts.

Run:  python train.py
"""

import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
TEST_SIZE = 0.2
TARGET_RECALL = 0.98     # share of malignant cases we want to catch
N_BOOTSTRAP = 2000

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORT_DIR = ROOT / "reports"
EXAMPLE_PATH = ROOT / "examples" / "sample_input.csv"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data():
    """Return features X and labels y, where y = 1 means MALIGNANT."""
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    # In the original dataset 0 = malignant and 1 = benign. We flip it so that
    # 1 = malignant, the class we must not miss ("positive" class).
    y = (data.target == 0).astype(int)
    y.name = "malignant"
    return X, y


def build_models():
    """The candidate models. Scale-sensitive models get a StandardScaler."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False),
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


# ----------------------------------------------------------------------------
# Model selection and threshold tuning
# ----------------------------------------------------------------------------
def compare_models(models, X_train, y_train):
    """5-fold cross-validation on the training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["roc_auc", "recall", "precision", "accuracy"]
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "roc_auc": s["test_roc_auc"].mean(),
            "roc_auc_std": s["test_roc_auc"].std(),
            "recall": s["test_recall"].mean(),
            "precision": s["test_precision"].mean(),
            "accuracy": s["test_accuracy"].mean(),
        })
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return table.reset_index(drop=True)


def choose_threshold(y_true, proba, target_recall=TARGET_RECALL):
    """
    Pick the highest threshold that still catches `target_recall` of malignant
    cases. A higher threshold means fewer false alarms, so the highest
    threshold meeting the recall goal gives the best precision at that goal.
    Returns (threshold, table of recall/precision for every threshold).
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "recall": recall_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
        })
    table = pd.DataFrame(rows)
    ok = table[table["recall"] >= target_recall]
    if len(ok):
        chosen = float(ok["threshold"].max())
    else:  # goal not reachable: fall back to the most sensitive threshold
        chosen = float(table.loc[table["recall"].idxmax(), "threshold"])
    return chosen, table


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(y_true, proba, threshold):
    """All the metrics for one decision threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def bootstrap_ci(y_true, proba, threshold, n=N_BOOTSTRAP, seed=SEED):
    """
    95% bootstrap confidence intervals for recall and precision.
    The test set is small, so a single number can be misleading: the interval
    shows how much the result could change with a different sample of patients.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_all = (np.asarray(proba) >= threshold).astype(int)
    recalls, precisions = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], pred_all[idx]
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {"recall_95ci": interval(recalls), "precision_95ci": interval(precisions)}


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def plot_threshold_tradeoff(table, chosen, target_recall, path):
    plt.figure(figsize=(8, 5))
    plt.plot(table["threshold"], table["recall"], label="Recall (malignant cases caught)", color="#E45756", linewidth=2)
    plt.plot(table["threshold"], table["precision"], label="Precision (flags that are truly malignant)", color="#4C78A8", linewidth=2)
    plt.axhline(target_recall, color="grey", linestyle=":", label=f"Recall goal ({target_recall:.0%})")
    plt.axvline(chosen, color="black", linestyle="--", label=f"Chosen threshold = {chosen:.2f}")
    plt.xlabel("Decision threshold (flag a case if malignancy probability ≥ threshold)")
    plt.ylabel("Score (cross-validated, training data)")
    plt.title("Choosing the threshold: catching cancers vs false alarms")
    plt.ylim(0.5, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(y_test, proba, threshold, path):
    pred = (proba >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Benign", "Malignant"], cmap="Blues", colorbar=False
    )
    plt.title(f"Test set confusion matrix (threshold = {threshold:.2f})")
    plt.ylabel("Actual diagnosis")
    plt.xlabel("Model decision")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_roc(y_test, proba, name, path):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#4C78A8", linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
    plt.xlabel("False positive rate (benign cases wrongly flagged)")
    plt.ylabel("True positive rate (malignant cases caught)")
    plt.title("ROC curve on the held-out test set")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_importance(model, X_test, y_test, path):
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED
    )
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False).head(10)
    imp = imp.iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#72B7B2", capsize=3)
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.title("Top 10 most influential measurements (permutation importance)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return imp.iloc[::-1]


# ----------------------------------------------------------------------------
# Example input file for predict.py
# ----------------------------------------------------------------------------
def write_example_input(X_test, y_test, path):
    """Save 6 held-out cases (3 malignant, 3 benign) so predict.py can be tried."""
    malignant = list(y_test[y_test == 1].index[:3])
    benign = list(y_test[y_test == 0].index[:3])
    order = [malignant[0], benign[0], malignant[1], benign[1], malignant[2], benign[2]]
    sample = X_test.loc[order].copy()
    sample.insert(0, "actual", ["malignant" if y_test.loc[i] == 1 else "benign" for i in order])
    sample.insert(0, "sample_id", [f"case_{k + 1}" for k in range(len(order))])
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. Data and held-out test set
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Cases: {len(X)} total | {len(X_train)} train | {len(X_test)} test")
    print(f"Malignant: {int(y.sum())} | Benign: {int((y == 0).sum())}")

    # 3. Compare models (training data only)
    print("\nComparing models with 5-fold cross-validation...")
    comparison = compare_models(build_models(), X_train, y_train)
    comparison.round(4).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(comparison.round(3).to_string(index=False))

    best_name = comparison.loc[0, "model"]
    model = build_models()[best_name]
    print(f"\nSelected model (highest cross-validated ROC-AUC): {best_name}")

    # 4. Tune the decision threshold on cross-validated predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, tradeoff = choose_threshold(y_train, oof_proba)
    print(f"Chosen threshold: {threshold:.2f} (goal: catch >= {TARGET_RECALL:.0%} of malignant cases)")

    # 5. Final fit and single evaluation on the test set
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    default_metrics = evaluate(y_test, proba_test, 0.5)
    tuned_metrics = evaluate(y_test, proba_test, threshold)
    intervals = bootstrap_ci(y_test, proba_test, threshold)

    print("\nHELD-OUT TEST RESULTS")
    for label, m in (("Default threshold 0.50", default_metrics), (f"Tuned threshold {threshold:.2f}", tuned_metrics)):
        print(f"  {label}: recall {m['recall']:.1%} | precision {m['precision']:.1%} | "
              f"missed cancers {m['false_negatives']} | false alarms {m['false_positives']} | "
              f"accuracy {m['accuracy']:.1%}")
    lo, hi = intervals["recall_95ci"]
    print(f"  Recall 95% confidence interval (tuned): {lo:.1%} to {hi:.1%}")
    print(f"  ROC-AUC: {tuned_metrics['roc_auc']:.3f}")

    # 6. Charts, example file, saved model and metrics
    plot_threshold_tradeoff(tradeoff, threshold, TARGET_RECALL, REPORT_DIR / "threshold_tradeoff.png")
    plot_confusion(y_test, proba_test, threshold, REPORT_DIR / "confusion_matrix.png")
    plot_roc(y_test, proba_test, best_name, REPORT_DIR / "roc_curve.png")
    top_features = plot_importance(model, X_test, y_test, REPORT_DIR / "feature_importance.png")
    write_example_input(X_test, y_test, EXAMPLE_PATH)

    bundle = {
        "model": model,
        "model_name": best_name,
        "threshold": threshold,
        "feature_names": list(X.columns),
        "target_recall": TARGET_RECALL,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "cases": int(len(X)),
            "features": int(X.shape[1]),
            "malignant": int(y.sum()),
            "benign": int((y == 0).sum()),
            "train_cases": int(len(X_train)),
            "test_cases": int(len(X_test)),
            "test_malignant": int(y_test.sum()),
        },
        "selected_model": best_name,
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "cross_validation": comparison.round(4).to_dict(orient="records"),
        "test_default_threshold": default_metrics,
        "test_tuned_threshold": tuned_metrics,
        "test_tuned_confidence_intervals": intervals,
        "top_features": top_features[["feature", "importance"]].round(4).to_dict(orient="records"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "random_seed": SEED,
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics and charts to {REPORT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
