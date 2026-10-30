# Machine Learning-Based Analysis and Prediction of Employee Earnings Using MTA Payroll Data

An end-to-end machine learning workflow in Python that cleans the **MTA Employee Payroll: Beginning 2025** open dataset, explores it visually, predicts employee **Total Earnings** with four regression models, and segments employees with **K-means clustering**.

**Author:** Sahil Koundal – Computer Science and Engineering, Lovely Professional University, Jalandhar, India
**Contact:** sahilkoundal756@gmail.com

---

## Overview

Public-sector payroll data shows how compensation relates to pay rate, tenure, job title, and agency. This project:

1. Cleans and prepares the raw payroll records.
2. Explores the distribution of earnings and its relationship to other variables.
3. Builds and compares four regression models that predict total earnings.
4. Discovers natural employee segments using K-means clustering.

## Dataset

- **Source:** [MTA Employee Payroll: Beginning 2025](https://data.ny.gov) (New York State Open Data)
- **Target:** `Total Earnings`
- **Key columns:** Hourly Rate, Regular Pay, Overtime Pay, Cash Outs, Retro Pay, Other Pay, Start Date, Separation Date, Working Agency, Title, Department, Pay Basis

> The raw data file is not included in this repository. Download it from the link above and place it in the project's data folder.

## Methodology

**Cleaning**
- Strip column names, remove exact duplicates, coerce pay columns to numeric, parse dates.
- Treat the placeholder separation date `1900-01-01` as missing.
- Drop records with missing or negative `Total Earnings`.

**Feature engineering**
- `Start Year`, `Start Month`, `Tenure Years Approx` (reference date 31 Dec 2025, days / 365.25, clipped at 0), and `Has Separation Date`.

**Preprocessing and split**
- 80/20 train–test split, random seed 42.
- Numeric: median imputation + standardization. Categorical: most-frequent imputation + one-hot encoding.
- Pay components are excluded from regression predictors to avoid target leakage.

**Models**

| Model | Description |
| --- | --- |
| Simple Linear | Hourly Rate only (baseline) |
| Multiple Linear | All numeric + categorical features |
| Polynomial (deg. 2) | Hourly Rate and Tenure Years Approx |
| Random Forest | 100 trees, max depth 15, min 2 samples per leaf |

**Clustering:** K-means (k = 3, 10 initializations, seed 42) on Hourly Rate, Regular Pay, Overtime Pay, and Tenure, evaluated with the silhouette score.

## Results

### Regression (20% held-out test set)

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| **Random Forest** | **18,123.5** | **27,081.6** | **0.7153** |
| Multiple Linear | 21,005.7 | 29,802.9 | 0.6552 |
| Polynomial (deg. 2) | 28,153.2 | 39,394.8 | 0.3970 |
| Simple Linear | 30,098.4 | 42,271.0 | 0.3063 |

### Employee segments (K-means, k = 3)

| Cluster | Mean Total Earnings | Mean Hourly Rate | Mean Overtime Pay | Profile |
| --- | --- | --- | --- | --- |
| 0 | 137,675 | 45.10 | 35,749 | Mid-rate, overtime-driven |
| 1 | 148,542 | 73.41 | 2,791 | High-rate, base-pay driven |
| 2 | 7,716 | 37.27 | 71,895 | Low-earning (part-year / adjustment records) |

### Key takeaways
- The random forest performs best; most of the gain over the baseline comes from adding tenure and categorical attributes.
- Similar total earnings can come from very different combinations of base rate and overtime.

## Repository Structure

> Adjust this section to match your actual files and folders.

```
.
├── data/                # Raw MTA payroll file (not tracked)
├── output/              # Cleaned data, model comparison table, cluster labels, figures
├── mta_payroll_analysis.py
├── IEEE_MTA_Payroll_Report.docx
├── requirements.txt
└── README.md
```

## Getting Started

### Requirements
- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, seaborn

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run

```bash
python mta_payroll_analysis.py
```

The script saves the cleaned dataset, the model comparison table, the cluster-labelled records, and all figures to the output folder, so every number and plot in the report can be regenerated from the raw file.

## Limitations

- Tenure is approximated using a fixed reference date.
- k = 3 was fixed, not selected with the elbow or silhouette method.
- Random forest hyperparameters were set manually, not cross-validated.
- A single random split is used; repeated records for one person could make estimates optimistic.
- The dataset covers a single reporting period.

## Future Work

Cross-validated hyperparameter tuning, gradient boosting, regression on log-transformed earnings, automatic selection of k, permutation-based feature importance, and comparison with hierarchical or mixture-model clustering.

## Ethical Note

The data is public but contains individual-level compensation records. The models are intended for aggregate analysis (budgeting, anomaly detection), not for judging individual employees. Any use of residual-based anomaly flags should include human review.
