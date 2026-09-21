# Hospital Readmission Risk Prediction — Project Report

**Industry:** Healthcare  **Project Type:** Machine Learning – Classification  **Duration:** 2 Weeks

---

## 1. Problem Statement

Hospital readmissions within 30 days of discharge are a major concern for healthcare providers: they signal that a patient's condition was not fully stabilized at discharge, leading to poorer health outcomes and additional cost for both patient and provider.

This project builds a machine learning model that predicts the likelihood of a patient being readmitted within 30 days, based on demographics, diagnosis details, treatment information, and prior hospitalization history — and identifies the clinical/demographic factors most associated with risk, along with recommendations to improve post-discharge care planning.

## 2. Dataset

- **Source:** [Hospital Readmissions Dataset, Kaggle](https://www.kaggle.com/datasets/dubradave/hospital-readmissions)
- **Target variable:** `readmitted` (Yes / No)
- **Features:** patient age, length of stay, number of lab procedures/medications/procedures, prior outpatient/inpatient/emergency visit counts, medical specialty, up to three diagnosis codes, glucose and A1C test results, medication-change flag, diabetes-medication flag (plus `gender` and `discharge_disposition` in the notebook's development dataset).

**Two data sources are referenced in this repository and should not be confused:**

1. **Synthetic development dataset** (used for the notebook run summarized in this report) — 20,030 encounters, 19 columns, an imbalanced target (~18.5% readmitted). Built because the development environment could not reach Kaggle directly; it reproduces the same schema, realistic distributions, missing values, duplicates, and inconsistent labels so every cleaning/EDA/modeling step in the notebook is genuinely exercised.
2. **Real dataset provided in this repo** (`data/hospital_readmissions.csv`) — 25,000 encounters, 17 columns, **no missing values**, and a materially more balanced target (~47.0% readmitted vs. ~53.0% not readmitted). It also **does not include `gender` or `discharge_disposition`**, which the notebook currently uses.

**Action required before final submission:** re-run the notebook against the real file, dropping/guarding the two missing columns. Because the real data is far more balanced than assumed, the class-imbalance framing in Sections 7–8 below (and the `class_weight='balanced'` modeling choice) should be re-checked once re-run — the qualitative workflow stays valid, but exact metrics, the imbalance-driven metric argument, and feature importances will shift. This is documented as the primary limitation of the current results (Section 10).

## 3. Data Understanding

- Shape: 20,030 rows × 19 columns (7 numerical, 12 categorical, including the target).
- Missing values: `diag_3` (8.00%) and `medical_specialty` (5.01%) — realistic, since a secondary diagnosis or treating specialty is not always recorded.
- Duplicates: 30 exact duplicate rows found.
- Data quality issue: `gender` contained inconsistent casing (`MALE`, `female`, etc.), requiring cleanup.
- Target distribution: `readmitted` = **no: 81.5%, yes: 18.5%** — a class-imbalanced binary target typical of 30-day readmission problems, which shapes which evaluation metrics matter most (Section 8).

## 4. Statistical Analysis

Descriptive statistics (mean, median, mode, min/max, variance, std, quartiles, IQR) were computed for all seven numerical features, plus skewness and a correlation heatmap.

**Business interpretation:**
- `n_inpatient`, `n_emergency`, and `n_outpatient` are strongly right-skewed — most patients have zero prior visits, with a long tail of high-utilization patients. This small subgroup is disproportionately important for readmission risk.
- `time_in_hospital` (median ≈ 4 days) and `n_medications` (median ≈ 15) show wide IQRs, indicating varied case-mix — a one-size-fits-all discharge plan is unlikely to serve all patients equally well.
- The correlation heatmap shows no pair of numerical features strongly collinear (all |r| well below 0.7), so multicollinearity was not a concern for the linear model and no numerical feature needed to be dropped on that basis.

## 5. Exploratory Data Analysis (EDA)

**Univariate:** histograms of length of stay, lab procedures, medications, and prior-visit counts confirmed the right-skew noted above; a count plot of `readmitted` confirmed the ~81.5/18.5 class split; age-group, primary-diagnosis, and discharge-disposition distributions were also profiled.

**Bivariate / multivariate (vs. `readmitted`):**
- Boxplots show readmitted patients have visibly higher medians for `n_inpatient`, `n_emergency`, `time_in_hospital`, and `n_medications` — prior utilization is the strongest visual signal in the data.
- Patients with a **high A1C** result, or a **Circulatory** or **Diabetes** primary diagnosis, show a noticeably higher readmission rate than average.
- Patients discharged to **"Transferred to another facility"** or **"Home Health Care"** are readmitted more often than patients discharged straight **Home**, consistent with these patients being clinically more fragile at discharge.
- A scatter of prior inpatient vs. emergency visits (colored by readmission) shows the two move together, and both climb sharply for the readmitted group — a strong candidate pair for feature engineering.

## 6. Data Preprocessing

1. **Duplicates removed:** 30 exact duplicate rows dropped.
2. **Inconsistent labels fixed:** `gender` stripped and capitalized.
3. **Missing values handled:** `medical_specialty` → `"Missing/Other"`; `diag_3` → `"None"` (both treated as informative "not recorded" categories rather than dropped, since missingness itself may be meaningful).
4. **Outlier handling:** IQR-rule capping applied to `n_outpatient`, `n_inpatient`, `n_emergency` (their right-skew makes them the columns most likely to contain extreme values that could destabilize distance/gradient-based models).
5. **Target encoding:** `readmitted_flag` = 1 if `readmitted == 'yes'` else 0.
6. **Age encoding:** ordinal-encoded (`age_ordinal`), since age bands have a natural order.
7. **Nominal categoricals** (gender, diagnoses, medical specialty, tests, discharge disposition) one-hot encoded via `pd.get_dummies(drop_first=True)` after feature engineering.
8. **Train/test split:** 80/20, stratified on the target (Train: 16,000 rows / 55 features; Test: 4,000 rows; readmit rate preserved at 18.5% in both splits).
9. **Feature scaling:** `StandardScaler` applied for the distance/gradient-based models (Logistic Regression, KNN); tree-based models used the unscaled matrix.

## 7. Feature Engineering

Five engineered features were created, each with an explicit clinical rationale:

| Feature | Definition | Rationale |
|---|---|---|
| `prior_utilization_score` | `2×n_inpatient + 1.5×n_emergency + n_outpatient` | Combines correlated prior-visit counts into one weighted utilization signal (inpatient/emergency weighted higher as stronger risk indicators) |
| `high_risk_diag` | 1 if primary diagnosis ∈ {Circulatory, Diabetes} | These diagnoses showed the highest readmission rates in EDA |
| `abnormal_lab_flag` | 1 if A1C or glucose test = "high" | Signals poorer glycemic control at discharge |
| `medication_intensity` | `n_medications / time_in_hospital` | Normalizes medication burden by length of stay, capturing treatment aggressiveness relative to admission length |
| `complex_discharge` | 1 if discharge disposition ≠ "Home" | Non-home discharges showed elevated readmission rates in EDA |

All five features showed a visible separation between readmitted and non-readmitted groups on validation boxplots/group means, confirming they carry non-redundant signal beyond their raw source columns.

## 8. Model Development

Five classifiers were trained and compared, using `class_weight='balanced'` where supported to counter the ~81.5/18.5 imbalance:

- Logistic Regression (`max_iter=1000`, balanced)
- Decision Tree (`max_depth=6`, balanced)
- Random Forest (`n_estimators=300`, `max_depth=10`, balanced)
- AdaBoost (`n_estimators=200`)
- KNN (`n_neighbors=15`)

## 9. Model Evaluation

| Model | Train Acc. | Test Acc. | Train–Test Gap | ROC AUC | Recall | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.630 | 0.638 | −0.007 | **0.655** | **0.586** | 0.374 |
| Random Forest | 0.822 | 0.722 | 0.099 | 0.648 | 0.373 | 0.332 |
| AdaBoost | 0.815 | 0.816 | −0.001 | 0.647 | 0.018 | 0.034 |
| Decision Tree | 0.618 | 0.596 | 0.022 | 0.627 | 0.576 | 0.345 |
| KNN | 0.816 | 0.814 | 0.002 | 0.572 | 0.008 | 0.016 |

**Why Recall and ROC AUC matter more than Accuracy here:** with only ~18–19% of patients actually readmitted, a model that predicts "No" for everyone already scores ~80% accuracy while being clinically useless. The cost of a **false negative** (a truly at-risk patient not flagged, so no follow-up is scheduled and they are readmitted) is far higher than a **false positive** (a follow-up call placed to a patient who did not need it). This is why Recall and ROC AUC were weighted more heavily than raw Accuracy in model selection, and why `class_weight='balanced'` was used.

Note the high-accuracy/low-recall models (AdaBoost, KNN) illustrate exactly this trap: both post ~81–82% accuracy — close to the "always predict No" baseline — while catching under 2% of true readmissions (Recall = 0.018 and 0.008 respectively). They would be actively harmful if deployed, despite looking strong on accuracy alone.

## 10. Validating the Results & Final Model Selection

- **Overfitting check (train–test gap):** Random Forest shows the largest gap (0.099 accuracy points), an early warning sign it partially memorized the training set despite depth limiting. Logistic Regression, AdaBoost, and KNN all show near-zero gaps.
- **Final model selected: Logistic Regression.** It posts the **best ROC AUC (0.655)** and **best Recall (0.586)** of all five models, with a negligible train/test gap (best generalization of the group), and remains fully interpretable via its coefficients — a meaningful advantage for a model that will need clinical buy-in.
- On the held-out test set, Logistic Regression achieves: Accuracy 0.64, Recall 0.59 and Precision 0.27 for the "Yes" (readmitted) class, and Recall 0.65 / Precision 0.87 for "No" (4,000-row test set: 739 actually-readmitted, 3,261 not).
- **Model selection was deliberately not accuracy-driven** — three of the five models post higher raw accuracy than the selected model, but each does so by essentially defaulting to the majority class, which is clinically useless for this problem.

**Deploying this model in a hospital — how clinical staff would use it:**
1. At discharge, every patient is scored with a **readmission risk probability**, not a hard yes/no label.
2. Care coordinators sort the daily discharge list by predicted risk and prioritize phone-based follow-up or a home-health visit for the top risk band (e.g., top 15–20% of scores) within 48–72 hours.
3. The top feature drivers (prior inpatient/emergency visits, abnormal A1C, Circulatory/Diabetes diagnosis, complex discharge disposition) double as a discharge-planning checklist — e.g., confirming medication reconciliation and a follow-up appointment are booked before a high-risk patient leaves.
4. The score is treated as a **decision-support signal, not a diagnosis** — it flags who needs more attention; the final care plan still relies on clinical judgment.

## 11. Business Insights & Recommendations

**Highest-risk patient profile:** older patients (70+) with prior inpatient/emergency visits, an abnormal A1C or glucose result, a Circulatory or Diabetes primary diagnosis, and a discharge disposition other than straight-home (transfer or home health care).

**Strongest clinical drivers of readmission:** prior utilization (inpatient/emergency visit history) consistently outranks demographic factors — a recent hospitalization history is the clearest early-warning signal available at the point of discharge.

**Diagnoses/treatments linked to higher readmission:** Circulatory and Diabetes-related admissions; and, notably, patients whose diabetes medication was **not** changed despite an abnormal lab result — suggesting some treatment plans were not adjusted before discharge.

**Recommended post-discharge interventions:**
1. **Tiered follow-up program** — automatic 48–72 hour follow-up call for patients in the "High" prior-utilization tier; a nurse home visit for patients who are also elderly with an abnormal A1C.
2. **Medication reconciliation checkpoint** — flag high `medication_intensity` patients (many medications relative to a short stay) for pharmacist review before discharge.
3. **Diabetes-focused care pathway** — patients with high A1C/glucose who leave without a medication change should trigger an automatic endocrinology/primary-care follow-up within a week.
4. **Discharge-disposition monitoring** — track patients transferred to another facility or discharged with home health care more closely, and audit the current transition-of-care handoff for this group.
5. **Care-team dashboard** — feed the model's daily risk scores into a dashboard so coordinators can prioritize limited follow-up capacity toward the patients who need it most.

## 12. Limitations

- **Dataset mismatch (primary limitation):** results above were generated on a synthetic stand-in dataset (~18.5% readmitted, 19 columns) built because the development environment could not reach Kaggle. The real dataset now available in this repo (`data/hospital_readmissions.csv`) has ~47.0% readmitted and lacks the `gender`/`discharge_disposition` columns. The notebook should be re-run end-to-end on the real file before the reported metrics, feature importances, or the imbalance-driven metric argument are treated as final — the workflow itself does not need to change.
- The engineered risk signal is a simplification of real clinical risk; true readmission drivers also depend on social determinants of health (housing, caregiver support, insurance) not present in this dataset.
- The model captures **association, not causation** — e.g., "home health care" patients are not at higher risk *because* of the home health care itself, but because they were already sicker at discharge, which is why they were assigned home health care in the first place.
- Class imbalance in the development run means Precision for the "readmitted" class is modest (0.27); this was treated as an accepted trade-off given the higher cost of missing a true readmission, but should be re-evaluated once the more balanced real dataset is used.

## 13. Conclusion

This project implemented a complete, end-to-end classification workflow — from raw data understanding and cleaning, through statistical analysis, EDA, feature engineering, multi-model comparison, evaluation, and business translation — to predict 30-day hospital readmission risk. Prior hospitalization utilization, glycemic control indicators, and discharge disposition emerged as the strongest, most actionable risk signals, giving hospital care teams a concrete, interpretable basis (via the selected Logistic Regression model) for prioritizing post-discharge follow-up. The next step is to re-run the pipeline on the real provided dataset and confirm these findings hold under its different class balance and feature set.
