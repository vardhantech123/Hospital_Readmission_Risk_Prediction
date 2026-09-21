# Hospital Readmission Risk Prediction

A machine learning classification project that predicts whether a patient will be **readmitted to hospital within 30 days of discharge**, using demographic, diagnosis, treatment, and prior-utilization data — and translates the results into concrete post-discharge care recommendations.

> Healthcare · Machine Learning · Classification | Individual major project

---

## Problem Statement

Hospital readmissions within 30 days are costly and often signal that a patient's condition was not fully stabilized at discharge. This project builds a model that scores each patient's readmission risk so that care teams can prioritize follow-up (calls, home visits, medication review) for the patients most likely to bounce back.

## Repository Contents

| File | Description |
|---|---|
| `Hospital_Readmission_Risk_Prediction.ipynb` | End-to-end notebook: data understanding → statistics → EDA → preprocessing → feature engineering → modeling → evaluation → business insights |
| `data/hospital_readmissions.csv` | Patient encounter dataset (see [Dataset](#dataset)) |
| `Project_Report.md` | Full written report — methodology, findings, model comparison, and recommendations |
| `README.md` | This file |

## Dataset

- **Source:** [Hospital Readmissions Dataset — Kaggle](https://www.kaggle.com/datasets/dubradave/hospital-readmissions)
- **Provided file:** `data/hospital_readmissions.csv` — 25,000 patient encounters, 17 columns (age, length of stay, lab procedures, medications, prior outpatient/inpatient/emergency visits, medical specialty, up to three diagnoses, glucose/A1C test results, medication change flag, diabetes medication flag, and the `readmitted` target).
- **Target:** `readmitted` (`yes` / `no`)

> **⚠️ Note on the notebook's current data source:** the notebook in this repo was originally developed and run against a synthetic dataset generated to match the Kaggle schema (built because the development environment had no internet access to Kaggle). The real `hospital_readmissions.csv` now included in this repo has a **different class balance (~47% readmitted vs. the synthetic ~18–19%) and two fewer columns** (`gender` and `discharge_disposition` are not present in the real file). **Before submitting/using this project, re-run the notebook end-to-end against the real CSV** — see [Steps to Reproduce](#steps-to-reproduce) and `Project_Report.md § Limitations` for full details. The workflow, code structure, and modeling approach do not need to change; only the two columns referenced above (`gender`, `discharge_disposition`) must be dropped or made optional, and all metrics/plots should be regenerated.

## Approach

1. **Data Understanding** — shape, dtypes, missing values, duplicates, unique-value sanity checks
2. **Statistical Analysis** — mean/median/mode, variance, std, quartiles, IQR, skewness, correlation heatmap, with business interpretation
3. **Exploratory Data Analysis** — univariate, bivariate, and multivariate analysis of clinical/demographic variables against readmission
4. **Data Preprocessing** — duplicate removal, missing-value imputation, category-label cleanup, IQR-based outlier capping, target/age encoding
5. **Feature Engineering** — `prior_utilization_score`, `high_risk_diag`, `abnormal_lab_flag`, `medication_intensity`, `complex_discharge`
6. **Model Development** — Logistic Regression, Decision Tree, Random Forest, AdaBoost, KNN (all with `class_weight='balanced'` where supported, to address class imbalance)
7. **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC AUC, confusion matrices, ROC curves; train/test gap analysis for over/underfitting
8. **Business Insights & Recommendations** — highest-risk patient profiles, strongest clinical drivers, and a tiered post-discharge intervention plan

## Results (synthetic-data run — see note above)

| Model | Train Acc. | Test Acc. | ROC AUC | Recall | F1 |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.630 | 0.638 | **0.655** | **0.586** | 0.374 |
| Random Forest | 0.822 | 0.722 | 0.648 | 0.373 | 0.332 |
| AdaBoost | 0.815 | 0.816 | 0.647 | 0.018 | 0.034 |
| Decision Tree | 0.618 | 0.596 | 0.627 | 0.576 | 0.345 |
| KNN | 0.816 | 0.814 | 0.572 | 0.008 | 0.016 |

- **Selected model:** Logistic Regression — highest ROC AUC and Recall, near-zero train/test gap (best generalization), and fully interpretable coefficients for clinical use.
- **Why not Accuracy?** With only ~18–19% of patients actually readmitted, a model that always predicts "No" scores ~80% accuracy while catching zero at-risk patients. Recall and ROC AUC better reflect the real business cost of a missed readmission.
- Full metrics, confusion matrices, and ROC curves are in the notebook; full discussion is in `Project_Report.md`.

## Key Business Insights

- **Highest-risk profile:** patients aged 70+ with prior inpatient/emergency visits, an abnormal A1C or glucose result, a Circulatory or Diabetes primary diagnosis, and a non-home discharge disposition.
- **Strongest driver:** prior hospital utilization (inpatient/emergency visit history) outranks demographic factors as the clearest early-warning signal available at discharge.
- **Actionable finding:** patients with abnormal labs whose diabetes medication was *not* changed before discharge are a flagged gap in current care planning.

## Recommendations

1. Tiered 48–72 hour follow-up program based on risk score
2. Pharmacist medication-reconciliation review for high medication-intensity patients
3. Automatic diabetes-focused follow-up when labs are abnormal and medication wasn't adjusted
4. Closer tracking of patients discharged to transfer/home-health-care
5. A care-team dashboard surfacing daily risk scores to prioritize limited follow-up capacity

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · Jupyter Notebook

## Steps to Reproduce

```bash
git clone <this-repo-url>
cd hospital-readmission-risk-prediction
pip install -r requirements.txt   # numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, jupyter
jupyter notebook Hospital_Readmission_Risk_Prediction.ipynb
```

Run all cells top to bottom. The notebook loads `data/hospital_readmissions.csv` by default — replace it with the real Kaggle file (adjusting for the `gender`/`discharge_disposition` column difference noted above) to reproduce results on real data.

## Limitations

- Current results were generated on a synthetic stand-in dataset; re-run against the real data before relying on the reported metrics (see note above).
- The model captures association, not causation, and does not include social determinants of health (housing, caregiver support, insurance) that also affect readmission.
- Given class imbalance, precision on the "readmitted" class is modest — an accepted trade-off given the higher cost of a missed readmission.

## Author

Individual Major Project — Machine Learning Classification, Healthcare track.

## License

For academic/educational use. Dataset licensing follows the terms of the original [Kaggle dataset](https://www.kaggle.com/datasets/dubradave/hospital-readmissions).
