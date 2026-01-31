# 🔍 Credit Card Fraud Detection (Streamlit App)

**Streamlit Web App for Predicting Credit Card Fraud using ML Models**

This project presents an interactive Streamlit dashboard that allows users to upload transaction data and detect fraudulent records using trained machine learning models (Random Forest, XGBoost, Logistic Regression). The goal is to support fraud analysts and data teams with fast, explainable, and intuitive predictions.

---

## 📂 Project Files

- `streamlit_app.py` — Main Streamlit interface
- `final_randomforest.pkl` — Trained Random Forest model
- `final_xgboost.pkl` — Trained XGBoost model
- `final_logisticregression.pkl` — Trained Logistic Regression model
- `predictions.csv` — Model predictions from test data
- `feature_importance.csv` — Sorted feature importance scores
- `shap_values.csv` — SHAP values for explainability
- *(Optional)* `model_training.py` or `utils.py` — Scripts used for model training

---

## 📊 Dataset

- **Source**: Anonymized credit card transactions
- **Features**: V1 to V28 (PCA components), Time, Amount
- **Target**: `Fraud_Prediction` (1 = Fraudulent, 0 = Legitimate)
- **Samples**: 284,807 records with only 492 fraudulent cases (highly imbalanced)

---

## 🎯 Objectives

- Train multiple models to detect fraud:
  - Random Forest
  - XGBoost
  - Logistic Regression
- Provide an **interactive web app** for users to:
  - Upload `.csv` files
  - Get fraud predictions
  - View feature importance + SHAP visualizations
  - Download results

---

## 🧠 Modeling Approach

- **Models**:
  - `RandomForestClassifier` (`sklearn`)
  - `XGBClassifier` (`xgboost`)
  - `LogisticRegression` (`sklearn`)
- **Feature Engineering**:
  - Added `Hour` column from `Time`
  - Selected top features using Mutual Information, Pearson Correlation, RFE
- **Evaluation**:
  - Accuracy, ROC AUC, F1 Score, Confusion Matrix
  - SHAP summary plots for explainability

---

## 🖥️ Streamlit App Features

- 📂 Upload new transaction CSV
- 📈 Model selection (dropdown)
- 🧾 View prediction summary
- 📊 Download results with fraud labels
- 🧠 Feature importance table
- 🌈 SHAP value visualizations
- 📥 Export: `predictions.csv`, `shap_values.csv`, `feature_importance.csv`

---

## 🧪 Technologies Used

- Python 3.10+
- pandas, numpy
- scikit-learn
- xgboost
- shap
- matplotlib / seaborn
- streamlit

---

## 🚀 How to Run (Optional)

If you'd like to run locally:

```bash
conda activate fraudenv
cd path/to/app/
streamlit run streamlit_app.py
