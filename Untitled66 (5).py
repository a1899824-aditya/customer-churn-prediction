#!/usr/bin/env python
# coding: utf-8

# In[1]:


# File: eda_dataset_loading.py
import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/Aditya Venugopalan/Downloads/creditcard.csv")  


print(f"Dataset shape: {df.shape}")
print(df.head())



# In[2]:


# Check column types and missing values
print("\nColumn Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ###  Dataset Overview
# 
# - The dataset contains **284,807** credit card transactions across **31 columns**.
# - Features include:
#   - `V1–V28`: Anonymized PCA components
#   - `Time`: Seconds since the first transaction
#   - `Amount`: Transaction amount in Euros
#   - `Class`: Target variable (0 = Non-fraud, 1 = Fraud)
# - No missing values or non-finite values were found.
# - Dataset is highly imbalanced — only **492 out of 284,807 transactions (~0.17%)** are fraudulent.
# 

# In[3]:


#  Fraud / Non-Fraud Distribution

import seaborn as sns
import matplotlib.pyplot as plt

# Class balance
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()

# Count values
print(df['Class'].value_counts())
print(f"Fraud %: {round(df['Class'].mean() * 100, 4)}%")


# ### ⚖️ Class Distribution
# 
# - A count plot of the `Class` column confirms **severe class imbalance**:
#   - 284,315 transactions are legitimate (Class = 0)
#   - 492 transactions are fraudulent (Class = 1)
# - This imbalance (~0.17% fraud) will require careful handling during model training (e.g. resampling or specialized metrics like Precision-Recall).
# 

# In[4]:


import numpy as np

# Check for non-finite values
print("Number of non-finite values in Amount column:")
print(np.isfinite(df['Amount']).sum(), "/", len(df))

# Show rows with non-finite Amount values
invalid_rows = df[~np.isfinite(df['Amount'])]
print(invalid_rows)


# In[5]:


sns.histplot(data=df, x="Amount", hue="Class", bins=100, kde=True)
plt.title("Transaction Amount Distribution by Class")
plt.xlabel("Transaction Amount")
plt.show()


# ### Transaction Amount Distribution
# 
# - Most transactions are **below €200**, with a few large outliers up to €25,000.
# - The distribution is **heavily right-skewed** (long tail).
# - Fraudulent transactions are present across different amounts, but not concentrated at any specific value.
# - Log-scaling may help normalize this distribution for model input.
# 

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# ###  Correlation Heatmap
# 
# - The correlation matrix shows **minimal correlation between features**, which is expected due to PCA.
# - Some PCA components have **moderate correlations with the target**:
#   - `V11` (+0.15), `V4` (+0.13), `V2` (+0.09)
#   - `V14`, `V17`, and `V10` show **negative correlation** with fraud.
# - These features may carry useful signal for fraud detection models.
# 

# In[7]:


# Show features most correlated with fraud label
corr_with_class = corr_matrix['Class'].sort_values(ascending=False)
print("Top correlations with 'Class':\n", corr_with_class.head(10))
print("\nBottom correlations with 'Class':\n", corr_with_class.tail(10))


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Automatically select V1 to V28
pca_features = [f'V{i}' for i in range(1, 29)]

# Plot in groups of 4
for i in range(0, len(pca_features), 4):
    batch = pca_features[i:i+4]
    
    plt.figure(figsize=(16, 10))
    
    for j, col in enumerate(batch):
        plt.subplot(2, 2, j+1)
        sns.kdeplot(data=df, x=col, hue='Class', fill=True, common_norm=False, alpha=0.5)
        plt.title(f"{col} Distribution by Class")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# ###  PCA Component Distribution Analysis
# 
# - We examined the distribution of PCA components `V1–V28` for fraud vs non-fraud transactions using KDE plots.
# - Several components show **clear distribution shifts** between fraud and non-fraud:
#   - For example, `V4`, `V11`, and `V17` show distinct shapes or peaks.
# - This indicates these components may help models differentiate fraudulent behavior, even though their original meanings are unknown.
# 

# ###  Summary of EDA Insights
# 
# - The dataset is clean, with no missing or invalid values.
# - Fraud is extremely rare (~0.17%), requiring resampling or special metrics.
# - Some PCA components exhibit different distributions for fraud, showing potential for predictive power.
# - Feature scaling and handling class imbalance will be critical in the next steps.
# 

# In[9]:


df['Hour'] = (df['Time'] // 3600) % 24
sns.histplot(data=df, x='Hour', hue='Class', bins=24, multiple='stack')


# In[10]:


fraud = df[df['Class']==1]

sns.histplot(data=fraud, x='Hour', bins=24)
plt.title("Fraud by hours (Transactions)")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Count")
plt.show()


# ###  Time-Based Transaction Trends
# 
# - Most transactions occur between 9 AM and 11 PM, consistent with human activity patterns.
# - Very few transactions happen during early morning hours.
# - Due to class imbalance, fraud transactions are not visibly distinguishable in the full plot.
# - Thats why explore fraud-only time patterns separately to detect time-based fraud behavior.
# 

# ##### Pre - Processing Data

# In[11]:


missing_counts = df.isnull().sum()

missing = missing_counts[missing_counts >0]

print("Missing values per column:")
print(missing)


# ### Missing Values Check
# 
# - We checked for missing values in all 31 columns.
# -  No missing values were found.
# - This confirms that the dataset is complete and does not require imputation.
# 

# In[12]:


import numpy as np 

non_finite = df[~np.isfinite(df)].any()

print("Columns containing non-finite values:\n")
print(non_finite[non_finite == True])


# ###  Non-Finite Value Check (NaN, inf, -inf)
# 
# - We scanned all columns for non-finite values like NaN, +inf, and -inf.
# - None were found, confirming data integrity.
# - This ensures clean input for ML models and transformations.
# 

# In[13]:


from scipy.stats import zscore

# Select numerical features (excluding 'Class')
numeric_cols = df.select_dtypes(include=[np.number]).drop('Class', axis=1)

# Compute Z-score
z_scores = np.abs(zscore(numeric_cols))

# Flag outliers: Z > 3
outliers = (z_scores > 3).sum()

# Print count of outliers per feature
print("Outlier count per feature (Z > 3):\n")
print(outliers.sort_values(ascending=False))


# ###  Outlier Summary (Z-score > 3)
# 
# - Features such as `V27`, `V6`, `V20`, and `Amount` show high outlier counts.
# - These may reflect genuine anomalies and are **not removed** at this stage.
# - Fraud detection benefits from detecting these extremes rather than cleaning them.
# 

# In[14]:


df_model = df.copy()
df_model['Hour'] = (df_model['Time'] // 3600) %24


# In[15]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_model['Amount_scaled'] = scaler.fit_transform(df_model[['Amount']])


# In[16]:


# Keep 'Class' (target) as the last column
columns = [col for col in df_model.columns if col != 'Class'] + ['Class']
df_model = df_model[columns]


# In[17]:


df_model


# In[18]:


from sklearn.model_selection import train_test_split

X = df_model.drop('Class', axis=1)
y = df_model['Class']

X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)

print("Before SMOTE:")
print("Training class counts:\n", y_train.value_counts())
print("Testing class counts:\n", y_test.value_counts())


# ###  Train-Test Split (Stratified)
# 
# - The dataset was split into training (70%) and testing (30%) sets using stratification to preserve class distribution.
# - This ensures that both fraud and non-fraud examples are represented in each set.
# - No resampling was done on the test set to maintain a realistic fraud ratio for evaluation.
# 

# In[19]:


from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to training set only
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
print("After SMOTE:")
print(y_train_resampled.value_counts())


# ###  Class Balancing with SMOTE
# 
# - Applied SMOTE to the training data to synthetically oversample the fraud class.
# - Increased the number of fraudulent transactions to match the non-fraud count.
# - Ensures the model sees enough fraud patterns during training without altering test distribution.
# 

# In[20]:


print("Shapes:")
print("X_train:", X_train_resampled.shape)
print("y_train:", y_train_resampled.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# In[21]:


# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train_resampled)


# In[22]:


# Predict fraud labels (0 = Not Fraud, 1 = Fraud)
y_pred = logreg.predict(X_test)

# Predict fraud probabilities (for ROC curve and thresholding)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]


# In[23]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC Score: {auc:.4f}")


# In[24]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Logistic Regression)")
plt.grid(False)
plt.show()


# In[25]:


from sklearn.metrics import roc_curve, RocCurveDisplay

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.title("ROC Curve (Logistic Regression)")
plt.grid(True)
plt.show()


# ### Confusion Matrix
# 
# | Actual \ Predicted | Not Fraud (0) | Fraud (1) |
# |--------------------|---------------|-----------|
# | Not Fraud (0)      | 85276         | 19        |
# | Fraud (1)          | 148           | 0         |
# 
# **Interpretation:**
# 
# - **True Positives (TP = 0)**: No fraud transactions correctly detected.
# - **False Negatives (FN = 148)**: All frauds were missed.
# - **True Negatives (TN = 85,276)**: Non-fraud transactions correctly identified.
# - **False Positives (FP = 19)**: A few legitimate transactions were flagged as fraud.
# 
# **ROC Curve Insights:**
# - AUC ≈ 0.59 → very weak classifier performance.
# - ROC curve lies close to the diagonal — classifier is not able to distinguish well.
# 

# In[26]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
features_names = X_train.columns

rf_importances = pd.DataFrame({
    'Feature': features_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

from IPython.display import display
display(rf_importances.head(15))


# In[27]:


from IPython.display import display

# Show full feature importance list
display(rf_importances)


# In[28]:


# Plot top 15 feature importances
top_features = rf_importances.head()
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[29]:


from sklearn.metrics import classification_report , confusion_matrix, ConfusionMatrixDisplay, roc_auc_score , roc_curve, RocCurveDisplay

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))

fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_proba_rf)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Not Fraud", "Fraud"])
disp_rf.plot(cmap="Greens")
plt.title("Confusion Matrix (Random Forest)")
plt.grid(False)
plt.show()

# ROC Curve
print(classification_report(y_test, y_pred_rf))
roc_display_rf = RocCurveDisplay(fpr=fpr_rf, tpr=tpr_rf)
roc_display_rf.plot()
plt.title("ROC Curve (Random Forest)")
plt.grid(True)
plt.show()


# In[30]:


from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt

# Now calculating MI scores

mi_scores = mutual_info_classif(X_train, y_train, random_state = 42)

mi_series = pd.Series(mi_scores, index=X_train.columns)
mi_series = mi_series.sort_values(ascending =False)

plt.figure(figsize=(10,6))
mi_series.plot(kind='bar', color='skyblue')
plt.title("Mutual Information Scores per Feature")
plt.ylabel('MI Score')
plt.xlabel('Feature')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#  EXPLAINING EACH OUTPUT (Simple Summary)
# 1. Mutual Information (MI)
# 
# High Score → Strong relation with target (even non-linear).
# 
# We selected top 15 features like V17, V14, V12, etc.
# 
# Plotted bar chart showing MI scores ✅.

# In[31]:


# 1. Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 2. Concatenate features and target back to analyze full correlation
df_corr = pd.concat([X_train, y_train], axis=1)

# 3. Compute correlation matrix
corr_matrix = df_corr.corr()

# 4. Plot correlation heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Pearson Correlation Heatmap")
plt.tight_layout()
plt.show()


# In[32]:


import pandas as pd
import numpy as np

corr_matrix = df.corr()
cor_target = abs(corr_matrix['Class'])

top_features = cor_target.sort_values(ascending=False)[1:16]

print("Top features based on Pearson Correlation with target:\n")
print(top_features)

# Step 4: Get the list of feature names
selected_features_pearson = top_features.index.tolist()

# Step 5: Create new feature sets using only the selected features
X_train_pearson = X_train[selected_features_pearson]
X_test_pearson = X_test[selected_features_pearson]

# Now X_train_pearson and X_test_pearson are ready for model training


# 2. Pearson Correlation
# 
# Heatmap shows how strongly each feature is linearly related to the target.
# 
# Values close to ±1 are stronger.
# 
# You saw that features like V17, V12, V14 had good correlations.
# 
#  We'll extract the top 15 by absolute correlation (optional).

# In[33]:


# 1. Import libraries
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 2. Scale features between 0 and 1 (required for chi2)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Apply Chi-Square test
chi2_selector = SelectKBest(score_func=chi2, k=15)
chi2_selector.fit(X_train_scaled, y_train)

# 4. Get selected feature indices and names
selected_chi2_indices = chi2_selector.get_support(indices=True)
selected_chi2_features = X_train.columns[selected_chi2_indices]

# 5. Print selected features
print("Top 15 features selected by Chi-Square:")
print(selected_chi2_features)

# 6. Subset train/test for later model comparison
X_train_chi2 = X_train.iloc[:, selected_chi2_indices]
X_test_chi2 = X_test.iloc[:, selected_chi2_indices]


# In[34]:


chi2_scores = chi2_selector.scores_
chi2_series = pd.Series(chi2_scores, index=X_train.columns)
chi2_series = chi2_series.sort_values(ascending=False)

plt.figure(figsize=(10,6))
chi2_series.head(20).plot(kind='bar', color='orchid')
plt.title("Top Chi-Square Scores per Feature")
plt.ylabel("Chi-Square Score")
plt.xlabel("Feature")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 3. Chi-Square
# 
# Measures how much each feature's values differ from what’s expected by chance (good for discrete relationships).
# 
# Top 15 included things like Time, V3, V4, V7, V9, etc.
# 
# Needed MinMax scaling, which you did .

# In[35]:


# 1. Import libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 2. Set up Logistic Regression as the estimator
lr_estimator = LogisticRegression(max_iter=1000, random_state=42)

# 3. Initialize RFE to select top 15 features
rfe_selector = RFE(estimator=lr_estimator, n_features_to_select=15)

# 4. Fit RFE on training data
rfe_selector.fit(X_train, y_train)

# 5. Get the selected features
selected_rfe_indices = rfe_selector.get_support(indices=True)
selected_rfe_features = X_train.columns[selected_rfe_indices]

# 6. Print selected feature names
print("✅ Top features selected by RFE (Logistic Regression):\n", selected_rfe_features.tolist())

# 7. Reduce training and test sets to selected features
X_train_rfe = X_train.iloc[:, selected_rfe_indices]
X_test_rfe = X_test.iloc[:, selected_rfe_indices]


# 4. Recursive Feature Elimination (RFE)
# 
# Used Logistic Regression (or Random Forest earlier).
# 
# Iteratively removed least useful features.
# 
# Selected 15 features that best helped predict fraud.
# 
# Output: selected_rfe_features, X_train_rfe, X_test_rfe

# ###### Comparision of Feature selection

# Training on Full Feature Set (Set A)
# We'll start by training a RandomForestClassifier on the full original training data and then  evaluate it on the testing data later

# In[36]:


# Train & Evaluate on Full Feature Set

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# Training the model

rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)

y_pred_full = rf_full.predict(X_test)
y_proba_full = rf_full.predict_proba(X_test)[:,1]

print("Random Forest model perfomance based on full set")
print(classification_report(y_test, y_pred_full))

cm = confusion_matrix(y_test, y_pred_full)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix (Full Feature Set)")
plt.grid(False)
plt.show()

# 5. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_full)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.title("ROC Curve (Full Feature Set)")
plt.grid(True)
plt.show()

# 6. AUC Score
auc_score = roc_auc_score(y_test, y_proba_full)
print(f"AUC Score: {auc_score:.4f}")


# Traing  Model on Feature Set B (Top 15 Mutual Information Features)

# In[37]:


mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=X_train.columns)
mi_series = mi_series.sort_values(ascending=False)


# In[38]:


# 1. Select top 15 features based on MI
top_mi_features = mi_series.head(15).index

# 2. Subset training and test data
X_train_mi = X_train[top_mi_features]
X_test_mi = X_test[top_mi_features]


# In[39]:


rf_mi = RandomForestClassifier(n_estimators = 100, random_state=42)
rf_mi.fit(X_train_mi , y_train)

y_pred_mi = rf_mi.predict(X_test_mi)
y_proba_mi = rf_mi.predict_proba(X_test_mi)[:,1]

print(" Random Forest - MI Feature Set")
print(classification_report(y_test, y_pred_mi))

# 4. Confusion Matrix
cm_mi = confusion_matrix(y_test, y_pred_mi)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mi, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (MI Feature Set)")
plt.grid(False)
plt.show()

# 5. ROC Curve
fpr_mi, tpr_mi, _ = roc_curve(y_test, y_proba_mi)
roc_display = RocCurveDisplay(fpr=fpr_mi, tpr=tpr_mi)
roc_display.plot()
plt.title("ROC Curve (MI Feature Set)")
plt.grid(True)
plt.show()

# 6. AUC Score
auc_score_mi = roc_auc_score(y_test, y_proba_mi)
print(f"AUC Score (MI Feature Set): {auc_score_mi:.4f}")


# Trainimg Random Forest on Chi-Square Selected Features (Top 15  Chi-Square Selected Features)

# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# 1. Train model
rf_chi2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_chi2.fit(X_train_chi2, y_train)

# 2. Predictions
y_pred_chi2 = rf_chi2.predict(X_test_chi2)
y_proba_chi2 = rf_chi2.predict_proba(X_test_chi2)[:, 1]

# 3. Print classification report
print("📊 Random Forest - Chi-Square Feature Set")
print(classification_report(y_test, y_pred_chi2))

# 4. Confusion matrix
cm = confusion_matrix(y_test, y_pred_chi2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix (Chi-Square Features)")
plt.grid(False)
plt.show()

# 5. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba_chi2)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.title("ROC Curve (Chi-Square Features)")
plt.grid(True)
plt.show()

# 6. AUC Score
auc_score = roc_auc_score(y_test, y_proba_chi2)
print(f"AUC Score: {auc_score:.4f}")


# Trainimg Random Forest on RFE Selected Features (Top 15 RFE Selected Features)

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# 1. Train model
rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
rf_rfe.fit(X_train_rfe, y_train)

# 2. Predictions
y_pred_rfe = rf_rfe.predict(X_test_rfe)
y_proba_rfe = rf_rfe.predict_proba(X_test_rfe)[:, 1]

# 3. Print classification report
print("📊 Random Forest - RFE Feature Set")
print(classification_report(y_test, y_pred_rfe))

# 4. Confusion matrix
cm = confusion_matrix(y_test, y_pred_rfe)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix (RFE Features)")
plt.grid(False)
plt.show()

# 5. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba_rfe)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.title("ROC Curve (RFE Features)")
plt.grid(True)
plt.show()

# 6. AUC Score
auc_score = roc_auc_score(y_test, y_proba_rfe)
print(f"AUC Score: {auc_score:.4f}")


# Training Model on Union Feature Set

# In[42]:


# Assuming you already ran MI, Chi2, and RFE
# If not, rerun those cells and note the selected feature names

selected_mi_features = mi_series.head(15).index.tolist()         # from mutual_info_classif
selected_chi2_features = selected_chi2_features.tolist()          # from SelectKBest chi2
selected_rfe_features = selected_rfe_features.tolist()           # from RFE

# Confirm
print("MI features:", selected_mi_features)
print("Chi2 features:", selected_chi2_features)
print("RFE features:", selected_rfe_features)


# In[43]:


mi_features = set(selected_mi_features)
chi2_features = set(selected_chi2_features)
rfe_features = set(selected_rfe_features)


# In[44]:


# Union: all selected features
union_features = list(mi_features | chi2_features | rfe_features)
print(f"Union of features across MI, Chi2, and RFE ({len(union_features)}):\n", union_features)

X_train_union = X_train[union_features]
X_test_union = X_test[union_features]


# In[45]:


# 1. Train model
rf_union = RandomForestClassifier(n_estimators=100, random_state=42)
rf_union.fit(X_train_union, y_train)

# 2. Predictions
y_pred_union = rf_union.predict(X_test_union)
y_proba_union = rf_union.predict_proba(X_test_union)[:, 1]

# 3. Evaluation
print("📊 Random Forest - Union Feature Set")
print(classification_report(y_test, y_pred_union))

# 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_union)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Union Features)")
plt.grid(False)
plt.show()

# 5. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_union)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title("ROC Curve (Union Features)")
plt.grid(True)
plt.show()

# 6. AUC Score
auc_score = roc_auc_score(y_test, y_proba_union)
print(f"AUC Score: {auc_score:.4f}")


# In[46]:


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

# Store model results in a list of dicts
results = []

results.append({
    "Feature Set": "Full",
    "Accuracy": accuracy_score(y_test, y_pred_full),
    "F1 Score": f1_score(y_test, y_pred_full),
    "ROC AUC": roc_auc_score(y_test, y_proba_full)
})

results.append({
    "Feature Set": "Mutual Info (Top 15)",
    "Accuracy": accuracy_score(y_test, y_pred_mi),
    "F1 Score": f1_score(y_test, y_pred_mi),
    "ROC AUC": roc_auc_score(y_test, y_proba_mi)
})

results.append({
    "Feature Set": "Chi-Square (Top 15)",
    "Accuracy": accuracy_score(y_test, y_pred_chi2),
    "F1 Score": f1_score(y_test, y_pred_chi2),
    "ROC AUC": roc_auc_score(y_test, y_proba_chi2)
})

results.append({
    "Feature Set": "RFE (Top 15)",
    "Accuracy": accuracy_score(y_test, y_pred_rfe),
    "F1 Score": f1_score(y_test, y_pred_rfe),
    "ROC AUC": roc_auc_score(y_test, y_proba_rfe)
})

results.append({
    "Feature Set": "Union of Selected",
    "Accuracy": accuracy_score(y_test, y_pred_union),
    "F1 Score": f1_score(y_test, y_pred_union),
    "ROC AUC": roc_auc_score(y_test, y_proba_union)
})


# In[47]:


# Convert to DataFrame
comparison_df = pd.DataFrame(results)

# Display nicely
print("📊 Model Comparison Table:\n")
display(comparison_df.sort_values(by="F1 Score", ascending=False))


# In[48]:


import matplotlib.pyplot as plt

# Plot F1 Score comparison
plt.figure(figsize=(10, 6))
plt.bar(comparison_df["Feature Set"], comparison_df["F1 Score"], color='skyblue')
plt.title("F1 Score by Feature Set")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[49]:


# Lets import our model first

from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression

logreg = LogisticRegression(max_iter =1000, random_state=42)

# Now lets train the model based on Chi - Square features
logreg.fit(X_train_chi2,y_train)

#predicting the class labels and probabilities

y_pred_logreg = logreg.predict(X_test_chi2)
y_proba_logreg = logreg.predict_proba(X_test_chi2)[:,1]

# Finally evaluating the model performance 

from sklearn.metrics import classification_report , roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
print(" Logistic Regression Performance (Chi-Square Features):")
print(classification_report(y_test, y_pred_logreg))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba_logreg), 4))

# Step 6: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_logreg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Step 7: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba_logreg)
plt.plot(fpr, tpr, label="Logistic Regression (AUC = {:.4f})".format(roc_auc_score(y_test, y_proba_logreg)))
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid()
plt.show()




# In[50]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, ConfusionMatrixDisplay, roc_curve


# In[51]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

#  Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_chi2, y_train)

#  Predict
y_pred_xgb = xgb_model.predict(X_test_chi2)
y_proba_xgb = xgb_model.predict_proba(X_test_chi2)[:, 1]

# Evaluate

print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test,y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print("ROC AUC", roc_auc_score(y_test,y_proba_xgb))

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


# In[52]:


import pandas as pd

# Create updated model comparison data
comparison_data = [
    {
        "Feature Set": "Chi-Square (Top 15)",
        "Model": "Random Forest",
        "Accuracy": 0.9996,
        "F1 Score": 0.8664,
        "ROC AUC": 0.9274
    },
    {
        "Feature Set": "Chi-Square (Top 15)",
        "Model": "Logistic Regression",
        "Accuracy": 0.9984,
        "F1 Score": 0.8354,
        "ROC AUC": 0.9116
    },
    {
        "Feature Set": "Chi-Square (Top 15)",
        "Model": "XGBoost",
        "Accuracy": 0.9996,
        "F1 Score": 0.8567,
        "ROC AUC": 0.9267
    }
]

# Convert to DataFrame
comparison_df_advanced = pd.DataFrame(comparison_data)

# Display sorted by F1 score
print("📊 Updated Model Comparison Table:\n")
display(comparison_df_advanced.sort_values(by="F1 Score", ascending=False))


# In[53]:


import matplotlib.pyplot as plt

# Plot F1 scores
plt.figure(figsize=(10, 6))
plt.bar(comparison_df_advanced["Model"], comparison_df_advanced["F1 Score"], color="skyblue")
plt.title("F1 Score Comparison by Classifier (Chi-Square Features)")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0.8, 0.9)
plt.grid(True)
plt.tight_layout()
plt.show()


# Saving the model & Features

# In[54]:


import joblib

joblib.dump(rf_chi2,"final_rf.pkl")

joblib.dump(selected_chi2_features, "final.features.pkl")

print("Model and features saved")


# In[55]:


cd C:\Users\Aditya Venugopalan


# In[64]:


import streamlit as st
import pandas as pd
import joblib

# Load model + required columns
model = joblib.load("final_rf.pkl")
selected_features = joblib.load("final.features.pkl")

st.title("🕵️ Fraud Detection App")
st.write("### Expected columns for prediction:")
st.code(selected_features)

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Uploaded CSV:")
    st.dataframe(df.head())

    # Check + create Hour if possible
    if "Hour" not in df.columns:
        if "Time" in df.columns:
            df["Hour"] = (df["Time"] // 3600).astype(int)
            st.info("🕒 'Hour' column created from 'Time'.")
        else:
            st.error("⛔ 'Time' column missing. Cannot create 'Hour'.")
            st.stop()

    # 🔍 Re-check columns AFTER creating Hour
    df_columns_set = set(df.columns)
    missing = [col for col in selected_features if col not in df_columns_set]
    extra = [col for col in df.columns if col not in selected_features and col != 'Class']

    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()
    if extra:
        st.info(f"ℹ️ Extra columns (ignored): {extra}")

    # ✅ Filter + reorder columns
    df = df[selected_features]

    # 🧠 Debug: Confirm columns now match
    st.write("✅ Final columns used for prediction:")
    st.code(df.columns.tolist())

    if st.button("🔍 Predict Fraud"):
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        st.markdown("### 🔎 Prediction:")
        if pred == 1:
            st.error(f"🚨 FRAUD (Probability: {prob:.2f})")
        else:
            st.success(f"✅ NOT FRAUD (Probability: {prob:.2f})")


# In[ ]:




