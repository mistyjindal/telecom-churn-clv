# Telecom Customer Churn Prediction & CLV Analysis

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.5-blue?logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3.3-blue?logo=pandas)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete end-to-end machine learning project that **predicts customer churn** and **models Customer Lifetime Value (CLV)** for a US telecom company. Built on the IBM Telco Customer Churn dataset (7,042 customers, 50 features), this project goes from raw data all the way to actionable business segmentation.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline — 17 Steps](#-pipeline--17-steps)
- [Results](#-results)
- [Key Business Insights](#-key-business-insights)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🎯 Problem Statement

Telecom companies lose significant revenue every year to customer churn. Retaining an existing customer costs far less than acquiring a new one. This project answers three business questions:

1. **Which customers are likely to churn?** → Binary Classification
2. **How much is each customer worth?** → CLV Regression
3. **Which customers should we focus on first?** → 4-Quadrant Segmentation (CLV × Risk)

The final output is a prioritized list of customers with their churn probability, CLV, and recommended action — ready to feed into a CRM or retention campaign.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | IBM Telco Customer Churn Dataset |
| **File** | `telecom.3.csv` |
| **Raw Shape** | 7,043 rows × 50 columns |
| **After Cleaning** | 7,042 rows × 37 columns |
| **Target Column** | `Churn Label` (Yes → 1, No → 0) |
| **Churn Rate** | 26.5% |
| **Class Imbalance** | Handled with SMOTE on training data only |

### Feature Categories

| Category | Columns |
|---|---|
| **Demographics** | Gender, Age, Under 30, Senior Citizen, Married, Dependents |
| **Location** | Population (geographic cols dropped) |
| **Services** | Phone, Internet Type, Streaming TV/Movies/Music, Online Security, Online Backup, Device Protection, Tech Support, Unlimited Data |
| **Billing** | Monthly Charge, Total Charges, Total Revenue, Contract, Payment Method, Paperless Billing |
| **Engagement** | Tenure in Months, Offer, Number of Referrals, Satisfaction Score |
| **Engineered** | CLV, Service Count, Charges per GB, Is Long Term |

> ⚠️ **Leakage Prevention:** `Customer Status`, `Churn Category`, `Churn Reason`, `Churn Score`, and `CLTV` are dropped before any modeling step to prevent data leakage.

---

## 📁 Project Structure

```
telecom-churn-clv/
│
├── telecom_churn_clv.ipynb      ← Main notebook — all 17 steps
├── telecom.3.csv                ← Raw dataset
├── README.md                    ← This file
├── requirements.txt             ← All Python dependencies
│
├── outputs/                     ← Auto-created when notebook runs
│   ├── eda_plots.png            ← 6-panel EDA overview chart
│   ├── correlation_heatmap.png  ← Numeric feature correlations
│   ├── model_comparison.png     ← All models metric comparison
│   ├── roc_curves.png           ← ROC curves for all classifiers
│   ├── confusion_matrix.png     ← Best model confusion matrix
│   ├── feature_importance.png   ← Top 20 feature importances
│   ├── clv_analysis.png         ← CLV distribution & by contract
│   ├── clv_prediction.png       ← Predicted vs Actual CLV
│   ├── segment_scatter.png      ← CLV vs Churn Prob scatter
│   ├── churn_clv_output.csv     ← Per-customer predictions
│   └── segment_summary.csv      ← Aggregated segment stats
│
└── models/                      ← Saved model artifacts
    ├── churn_model.pkl           ← Trained Random Forest model
    └── scaler.pkl                ← Fitted StandardScaler
```

---

## 🔁 Pipeline — 17 Steps

```
Raw Data: 7,043 rows × 50 columns
          │
          ▼
Step 1  ── Imports & Setup
          pandas 2.3.3 | numpy 2.3.5 | sklearn | imbalanced-learn
          │
Step 2  ── Data Loading & Inspection
          Shape: (7043, 50) | 50 columns examined
          │
Step 3  ── Exploratory Data Analysis
          6-panel chart: Churn distribution, Tenure vs Churn,
          Monthly Charge vs Churn, Satisfaction Score vs Churn,
          Contract Type vs Churn, Monthly Charge distribution
          │
Step 4  ── Data Cleaning
          ├─ Fill nulls: Offer → "No Offer", Internet Type → "No Internet"
          ├─ Drop 1 null in Churn Label → 7,042 rows remain
          └─ Drop 13 leakage/irrelevant cols → 37 cols remain
          │
Step 5  ── Feature Engineering (4 new features)
          ├─ CLV = Monthly Charge × Tenure in Months
          ├─ Service_Count = sum of 8 add-on subscriptions
          ├─ charges_per_gb = Monthly Charge / (Avg GB + 1)
          └─ is_long_term = 1 if tenure > 24 months
          │
Step 6  ── Encoding
          ├─ Binary Yes/No → 1/0 (17 columns)
          ├─ Gender → Male=1, Female=0
          └─ One-hot encode: Contract, Internet Type, Payment Method, Offer
          │
Step 7  ── Correlation Heatmap
          Top correlations with Churn identified
          │
Step 8  ── Train/Test Split + SMOTE
          ├─ 80/20 stratified split
          ├─ StandardScaler applied
          └─ SMOTE balances training classes only
          │
Step 9  ── Model Training (4 Classifiers)
          Logistic Regression | Decision Tree |
          Random Forest | Gradient Boosting
          │
Step 10 ── Hyperparameter Tuning
          GridSearchCV on Random Forest (5-fold CV, ROC-AUC scoring)
          Best params: n_estimators, max_depth optimized
          │
Step 11 ── Model Evaluation
          Confusion Matrix | ROC Curves | Classification Report
          │
Step 12 ── Feature Importance
          Top 20 features from best Random Forest model
          │
Step 13 ── CLV Regression
          Ridge | Random Forest Regressor | Gradient Boosting Regressor
          │
Step 14 ── CLV Segmentation
          Customers split into Low / Medium / High CLV tertiles
          │
Step 15 ── 4-Quadrant Customer Segmentation
          CLV × Churn Probability matrix with scatter plot
          │
Step 16 ── Save Outputs
          CSV outputs + pkl model artifacts saved
          │
Step 17 ── Final Business Summary
          ════════════════════════════════════
          Dataset    : 7,042 customers
          Churn Rate : 26.5%
          Best Model : Random Forest (Tuned)
          Best AUC   : 0.9857
          Best F1    : 0.9339
          CLV R²     : 0.9991
          CLV RMSE   : $66
          High-Risk  : 648 customers
          CLV at Risk: $2,340,673
          ════════════════════════════════════
```

---

## 📈 Results

### Churn Classification — Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (Tuned)** | — | — | — | **0.9339** | **0.9857** |
| Gradient Boosting | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |

> 🏆 **Best Model: Random Forest (Tuned)** via GridSearchCV (5-fold CV, ROC-AUC scoring)
> AUC: **0.9857** | F1: **0.9339**

### CLV Regression — Model Comparison

| Model | R² | RMSE |
|---|---|---|
| **Random Forest Regressor** | **0.9991** | **$66** |
| Ridge Regression | 0.9991 | $66 |
| Gradient Boosting Regressor | 0.9991 | ~$68 |

> Extremely high R² (0.9991) confirms CLV = Monthly Charge × Tenure is a near-perfect linear relationship — the regression validates and generalizes this.

### Customer Segmentation Results

| Segment | Customers | Avg CLV | Avg Churn Prob | Action |
|---|---|---|---|---|
| 🔴 **High Value – High Risk** | 648 | High | > 50% | Immediate retention |
| 🟢 **High Value – Low Risk** | — | High | < 50% | Upsell & nurture |
| 🟡 **Low Value – High Risk** | — | Low | > 50% | Low-cost re-engagement |
| ⚪ **Low Value – Low Risk** | — | Low | < 50% | Standard service |

> 💰 **$2,340,673 in CLV is at immediate risk** from 648 high-value, high-risk customers

---

## 💡 Key Business Insights

- **26.5% of customers churn** — 1 in 4 customers leaves, representing substantial revenue loss
- **Month-to-month contracts** churn significantly more than annual or two-year contracts → incentivize longer plans with discounts
- **New customers (0–12 months)** are the highest-risk group → invest in onboarding and early engagement
- **Satisfaction score ≤ 2** is one of the strongest predictors of churn → set up proactive support triggers
- **Fiber Optic internet subscribers** churn at a higher rate than DSL or cable → investigate service quality
- **648 high-value customers** represent $2.34M in CLV at immediate risk → these are the top priority for retention budget
- **Service count matters** — customers subscribed to more add-ons (streaming, security, backup) churn less

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/telecom-churn-clv.git
cd telecom-churn-clv

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — Mac/Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

---

## 🚀 Usage

1. Place `telecom.3.csv` in the root project folder
2. Open `telecom_churn_clv.ipynb` in Jupyter
3. Run all cells from top to bottom:
   **Kernel → Restart & Run All**
4. All charts are auto-saved to `outputs/`
5. Customer predictions saved to `outputs/churn_clv_output.csv`
6. Trained model saved to `models/churn_model.pkl`

### Loading the saved model for predictions

```python
import joblib
import pandas as pd

# Load model and scaler
model  = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Predict on new data
X_new_scaled = scaler.transform(X_new)
churn_prob   = model.predict_proba(X_new_scaled)[:, 1]
```

---

## 📋 Requirements

```
numpy>=2.3.5
pandas>=2.3.3
matplotlib
seaborn
scikit-learn>=1.3.0
imbalanced-learn
joblib
jupyter
ipykernel
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Work

- [ ] Add **SHAP values** for explainability — show *why* each customer is predicted to churn
- [ ] Build a **Streamlit dashboard** for interactive churn risk scoring
- [ ] Deploy as a **REST API** using FastAPI so CRM systems can query predictions in real time
- [ ] Add **survival analysis** to model time-to-churn (not just whether, but when)
- [ ] Try **XGBoost / LightGBM** for potentially higher AUC
- [ ] Incorporate **geographic analysis** — churn patterns by city or population density
- [ ] Set up **MLflow** experiment tracking for model versioning and retraining

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Your Name**
📧 misty.jindal.2610@gmail.com
 [LinkedIn](https://www.linkedin.com/in/misty-jindal-7a2bb0288?utm_source=share_via&utm_content=profile&utm_medium=member_android))
 [GitHub](https://github.com/mistyjindal)

---

*Built with Python 3.13 | pandas 2.3.3 | NumPy 2.3.5 | scikit-learn | imbalanced-learn*
