# Breast Cancer Risk Classifier

A machine learning project that screens breast-mass measurements and **flags cases for clinician review**, built around the trade-off that matters most in cancer screening: *missing a malignant case is far worse than flagging a benign one for a second look.*

> ⚠️ **Educational prototype. Not a medical device.** It cannot diagnose cancer and must never be used to make decisions about a real patient's care. See [Limitations](#limitations).

**Final project: AI & ML Internship, Module 06 (Codomax Digital Solutions)**

---

## Why this project

Doctors and pathologists review large numbers of cases. A "second reader" model can help prioritise which ones need attention first. For this to be useful, the model has to be tuned for the right goal. Plain accuracy is not enough, because the two kinds of mistake have very different costs:

| Mistake | What it means | Cost |
|---|---|---|
| **Missed cancer** (false negative) | A malignant case is called benign | Delayed treatment, the worst outcome |
| **False alarm** (false positive) | A benign case is flagged | An extra check, stress, but no missed disease |

So this project does not just train a model. It **chooses a decision threshold that catches at least 98% of malignant cases**, then reports honestly what that costs in false alarms.

## Results

Evaluated **once** on a held-out test set of **114 cases** (42 malignant, 72 benign) that was never used for training, model selection or threshold tuning.

| | Default threshold (0.50) | **Tuned threshold (0.20)** |
|---|---|---|
| Malignant cases caught (recall) | 39 of 42 (92.9%) | **41 of 42 (97.6%)** |
| Missed cancers | 3 | **1** |
| False alarms (benign flagged) | 1 | 4 |
| Precision (flags that were truly malignant) | 97.5% | 91.1% |
| Accuracy | 96.5% | 95.6% |
| ROC-AUC | 0.996 | 0.996 |

95% bootstrap confidence intervals at the tuned threshold: **recall 92.1% to 100%**, precision 82.5% to 98.0%.

**Reading this honestly:** lowering the threshold caught 2 more cancers (3 missed → 1 missed) at the cost of 3 more false alarms (1 → 4). That is the trade-off working as intended, but these are very small counts, and the wide confidence interval shows the true recall could be noticeably lower or higher on other data.

![Threshold trade-off](reports/threshold_tradeoff.png)

| Confusion matrix (test set) | ROC curve (test set) |
|---|---|
| ![Confusion matrix](reports/confusion_matrix.png) | ![ROC curve](reports/roc_curve.png) |

### Model comparison (5-fold cross-validation, training data only)

| Model | ROC-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.996 | 0.953 | 0.977 | 0.974 |
| SVM (RBF, calibrated) | 0.995 | 0.959 | 0.977 | 0.976 |
| Gradient Boosting | 0.991 | 0.947 | 0.965 | 0.967 |
| Random Forest | 0.988 | 0.935 | 0.958 | 0.960 |

Logistic Regression had the highest cross-validated ROC-AUC. The top two models are very close (their standard deviations overlap), so the simple, explainable model is a sensible choice.

## How it works

1. **Data:** Breast Cancer Wisconsin (Diagnostic): 569 real cases (212 malignant, 357 benign), 30 numeric measurements of cell nuclei per case. No missing values.
2. **Split:** 80% training / 20% test, stratified so both sets keep the same malignant/benign proportions (`random_state=42`).
3. **Model selection:** four models compared with 5-fold cross-validation on the training data only.
4. **Threshold tuning:** the selected model's cross-validated probabilities on the training data are used to find the highest threshold that still catches at least 98% of malignant cases.
5. **Final test:** the model is fitted on the full training set and evaluated once on the untouched test set, with bootstrap confidence intervals.
6. **Outputs:** the saved model, metrics (`reports/metrics.json`) and charts.

The measurements (radius, texture, smoothness, concavity and so on, each as a mean, a standard error and a "worst" value) were computed from digitised images of fine-needle aspirates of breast masses.

## Quick start

Requires Python 3 (tested on Python 3.13).

```bash
git clone https://github.com/anushkaawari2006-afk/breast-cancer-risk-classifier.git
cd breast-cancer-risk-classifier
pip install -r requirements.txt

python train.py                              # re-train and regenerate reports (takes a few seconds)
python predict.py examples/sample_input.csv  # screen the example cases
python -m unittest discover -s tests -v      # run the 9 unit tests
```

The trained model is already included in `models/`, so `predict.py` works straight away.

### Using your own data

`predict.py` reads a CSV with the 30 measurement columns (same names as `examples/sample_input.csv`). Optional columns: `sample_id` (a label per row) and `actual` (the known diagnosis, shown for comparison).

```bash
python predict.py my_cases.csv --output results.csv
python predict.py my_cases.csv --threshold 0.30    # use a different threshold
```

It checks the input and reports clear errors for missing columns or non-numeric values.

### Example output

```
Model: Logistic Regression | decision threshold: 0.20

sample_id malignant_probability        decision    actual
   case_1                100.0% FLAG FOR REVIEW malignant
   case_2                  0.0%   Likely benign    benign
   case_3                 57.6% FLAG FOR REVIEW malignant
   case_4                  4.3%   Likely benign    benign
   case_5                 76.2% FLAG FOR REVIEW malignant
   case_6                 50.9% FLAG FOR REVIEW    benign

4 of 6 case(s) flagged for review.
```

These six cases come from the held-out test set. `case_6` is a **false alarm**: a benign case flagged at 50.9%. That is the cost of the cautious 0.20 threshold, and exactly the kind of error this design accepts in order to avoid missing cancers.

## Project structure

```
breast-cancer-risk-classifier/
├── train.py               # data, model comparison, threshold tuning, evaluation, saving
├── predict.py             # command-line tool to screen cases from a CSV file
├── requirements.txt
├── models/model.joblib    # trained model + feature names + decision threshold
├── examples/sample_input.csv
├── reports/               # metrics.json, model_comparison.csv and charts
├── tests/test_project.py  # 9 unit tests
├── LICENSE
└── README.md
```

## Model card

| | |
|---|---|
| **Intended use** | Learning project showing a complete ML workflow; demonstration of a screening-support idea. |
| **Not intended for** | Diagnosis, treatment decisions, or use on real patients of any kind. |
| **Model** | Logistic Regression with feature standardisation (scikit-learn). |
| **Input** | 30 numeric measurements per case. |
| **Output** | Probability of malignancy, and a decision ("FLAG FOR REVIEW" if probability ≥ 0.20). |
| **Training data** | 455 cases from the Wisconsin Diagnostic Breast Cancer dataset. |
| **Test data** | 114 held-out cases from the same dataset. |

## Limitations

Being clear about what this project **cannot** claim is part of doing ML responsibly:

- **One source, no external validation.** All data comes from a single study (University of Wisconsin Hospitals). Performance on other hospitals, equipment, populations or time periods is unknown, and would very likely be lower.
- **The data is relatively clean and easy.** Scores near 99% ROC-AUC are typical for this dataset. Real clinical data is messier.
- **Small test set.** 114 cases (42 malignant) means wide uncertainty, as the confidence intervals show.
- **It cannot be used on a raw image or a patient.** The 30 inputs come from image-analysis software run on a biopsy sample; this project starts after that step.
- **Probabilities are not guaranteed to be calibrated.** A "76%" is a model score, not a true 76% chance.
- **Feature importance is only a rough guide.** The importance chart in `reports/feature_importance.png` shows small scores with wide error bars, because many of the 30 measurements are strongly correlated. It should not be read as a medical finding.
- **No fairness analysis.** Subgroup performance (age, ethnicity and so on) was not examined because that information is not in the dataset.
- **Real clinical use would require** prospective validation, bias assessment, regulatory approval and clinician oversight.

## Future improvements

- Validate on a second, independent dataset
- Add calibration curves and per-case explanations (for example SHAP)
- Build a simple web interface for uploading a CSV
- Extend to image-based models on histopathology images
- Add automated retraining and monitoring checks

## Data source and licence

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The copy used here is the one bundled with scikit-learn (`sklearn.datasets.load_breast_cancer`).

The code in this repository is released under the [MIT License](LICENSE).

## Notes on reproducibility and safety

- All randomness uses `random_state=42`, so `python train.py` reproduces the numbers above on the same library versions.
- Developed and tested with Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3 and pandas 3.0.5. The saved model was created with scikit-learn 1.9.1; if you use another version and see a warning, run `python train.py` to regenerate it.
- Model files are loaded with `joblib`, which uses pickle. Only load model files from sources you trust.

## AI assistance

This project was developed with the help of an AI assistant (Claude by Anthropic) for writing and testing the code and drafting this documentation.

## Author

Anushka Awari
