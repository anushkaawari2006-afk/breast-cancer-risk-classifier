"""
predict.py - Screen new cases with the trained Breast Cancer Risk Classifier.

Input : a CSV file with the 30 measurement columns (see examples/sample_input.csv).
        Optional columns: "sample_id" (a label for each row) and
        "actual" (the known diagnosis, shown next to the prediction).
Output: for each row, the probability of malignancy and a decision.

Usage:
    python predict.py examples/sample_input.csv
    python predict.py my_cases.csv --output results.csv
    python predict.py my_cases.csv --threshold 0.3

EDUCATIONAL PROTOTYPE. Not a medical device. Not for diagnosis or treatment.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "model.joblib"

DISCLAIMER = (
    "NOTE: Educational prototype, not a medical device. It cannot diagnose cancer. "
    "Any real case must be assessed by a qualified clinician."
)


def load_bundle(path=DEFAULT_MODEL):
    """Load the saved model, its feature names and its decision threshold."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run 'python train.py' first.")
    return joblib.load(path)


def validate_input(df, feature_names):
    """Check the columns and values, and return the numeric feature table."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ValueError(f"Input is missing {len(missing)} required column(s): {shown}")

    features = df[feature_names].apply(pd.to_numeric, errors="coerce")
    bad = features.isnull().any(axis=1)
    if bad.any():
        positions = [int(i) + 2 for i in range(len(bad)) if bad.iloc[i]]  # +2: header row, 1-based
        raise ValueError(f"Missing or non-numeric values in file row(s): {positions[:10]}")
    return features


def predict_frame(bundle, df, threshold=None):
    """Return a results table with one row per input case."""
    thr = bundle["threshold"] if threshold is None else float(threshold)
    if not 0 < thr < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    features = validate_input(df, bundle["feature_names"])
    proba = bundle["model"].predict_proba(features)[:, 1]

    if "sample_id" in df.columns:
        ids = df["sample_id"].astype(str).tolist()
    else:
        ids = [f"row_{i + 1}" for i in range(len(df))]

    results = pd.DataFrame({
        "sample_id": ids,
        "malignant_probability": proba,
        "decision": ["FLAG FOR REVIEW" if p >= thr else "Likely benign" for p in proba],
    })
    if "actual" in df.columns:
        results["actual"] = df["actual"].astype(str).tolist()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Screen cases with the Breast Cancer Risk Classifier.")
    parser.add_argument("input_csv", help="CSV file with the 30 measurement columns")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the saved model")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the saved decision threshold (between 0 and 1)")
    parser.add_argument("--output", default=None, help="optional path to save the results as CSV")
    args = parser.parse_args(argv)

    try:
        bundle = load_bundle(args.model)
        df = pd.read_csv(args.input_csv)
        if df.empty:
            raise ValueError("The input file has no rows.")
        results = predict_frame(bundle, df, args.threshold)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    thr = bundle["threshold"] if args.threshold is None else args.threshold
    shown = results.copy()
    shown["malignant_probability"] = (shown["malignant_probability"] * 100).map("{:.1f}%".format)
    print(f"Model: {bundle['model_name']} | decision threshold: {thr:.2f}\n")
    print(shown.to_string(index=False))
    flagged = int((results["decision"] == "FLAG FOR REVIEW").sum())
    print(f"\n{flagged} of {len(results)} case(s) flagged for review.")
    print(DISCLAIMER)

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
