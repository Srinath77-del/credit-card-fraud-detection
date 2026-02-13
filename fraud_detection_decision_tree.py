# ============================================
# CREDIT CARD FRAUD DETECTION (Decision Tree)
# Works with your "fraud data set" ZIP or CSV
# Google Colab ready - Single Script
# ============================================

# 1. Imports
import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ------------------------------
# 2. Upload dataset (ZIP or CSV)
# ------------------------------
print("Please upload your dataset file (e.g. 'fraud data set' from Kaggle):")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(f"\nUploaded file: {filename}")

# ------------------------------
# 3. Load data (handles ZIP or CSV)
# ------------------------------
def load_dataset_from_uploaded(filename: str) -> pd.DataFrame:
    # Case 1: Direct CSV
    if filename.lower().endswith(".csv"):
        print("Detected CSV file. Reading directly...")
        return pd.read_csv(filename)

    # Case 2: ZIP file containing CSV (like your 'fraud data set')
    # Try opening as zip
    if zipfile.is_zipfile(filename):
        print("Detected ZIP file. Searching for CSV inside...")
        with zipfile.ZipFile(filename, 'r') as z:
            # List CSV files inside zip
            csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV file found inside the ZIP archive.")

            # Use the first CSV found (for Kaggle credit card fraud, it's 'creditcard.csv')
            inner_csv = csv_files[0]
            print(f"Found CSV inside ZIP: {inner_csv}")
            with z.open(inner_csv) as f:
                return pd.read_csv(f)

    # Otherwise, unsupported format
    raise ValueError("Please upload a CSV file or a ZIP file containing a CSV.")

# Load the dataframe
data = load_dataset_from_uploaded(filename)

print("\nFirst 5 rows of the raw data:")
print(data.head())
print("\nShape of raw data:", data.shape)

# ------------------------------
# 4. Basic info & class imbalance
# ------------------------------
if 'Class' not in data.columns:
    raise ValueError("Expected a 'Class' column for labels (0 = normal, 1 = fraud), but it was not found.")

print("\nClass distribution (0 = Normal, 1 = Fraud):")
print(data['Class'].value_counts())

fraud_percentage = data['Class'].value_counts()[1] / len(data) * 100
print("\nFraud percentage in data: {:.6f}%".format(fraud_percentage))

# ------------------------------
# 5. Data Cleaning
#    - Remove duplicates
#    - Handle missing values (if any)
# ------------------------------

# 5.1 Remove duplicate rows
before_dup = data.shape[0]
data = data.drop_duplicates()
after_dup = data.shape[0]
print(f"\nRemoved {before_dup - after_dup} duplicate rows (if any).")
print("New shape after removing duplicates:", data.shape)

# 5.2 Check for missing values
print("\nChecking for missing values in each column:")
null_counts = data.isna().sum()
print(null_counts)

# If there are missing values, we handle them:
if null_counts.sum() > 0:
    print("\nMissing values detected. Filling numerical columns with median and categorical with mode...")
    num_cols = data.select_dtypes(include=[np.number]).columns
    cat_cols = data.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)

    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
else:
    print("\nNo missing values detected. No imputation needed.")

# ------------------------------
# 6. Split features (X) and target (y)
# ------------------------------
X = data.drop('Class', axis=1)
y = data['Class']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# ------------------------------
# 7. Build & train Decision Tree model
# ------------------------------
dt_clf = DecisionTreeClassifier(
    criterion='gini',        # or 'entropy'
    max_depth=5,             # keep tree shallow for interpretability
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced', # important for highly imbalanced fraud data
    random_state=42
)

dt_clf.fit(X_train, y_train)
print("\nDecision Tree model trained successfully!")

# ------------------------------
# 8. Evaluate model
# ------------------------------
y_pred = dt_clf.predict(X_test)
y_proba = dt_clf.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Test Data):")
print(cm)

# Classification Report
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred, digits=4))

# ROC-AUC
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC (Test Data): {:.4f}".format(auc))

# ------------------------------
# 9. Plot ROC Curve
# ------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Card Fraud Detection (Decision Tree)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ------------------------------
# 10. Visualize Decision Tree (optional)
# ------------------------------
plt.figure(figsize=(22, 10))
plot_tree(
    dt_clf,
    feature_names=X.columns,
    class_names=['Normal (0)', 'Fraud (1)'],
    filled=True,
    rounded=True,
    fontsize=7
)
plt.title("Decision Tree (max_depth=5)")
plt.show()

print("\nPipeline completed: data loaded, cleaned, and model evaluated.")