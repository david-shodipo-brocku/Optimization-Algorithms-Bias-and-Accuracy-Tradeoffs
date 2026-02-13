# Annotated Code and Plain-English Results

This file contains a line-by-line commented version of every code cell from your notebook, followed by a short, non-technical explanation of the results produced by those cells.

---

### Cell: Libraries (setup)
```python
# Libraries
# Install packages when running interactively (run once in a fresh environment)
%pip install pandas numpy matplotlib seaborn scikit-learn

# Import numerical and tabular data packages
import numpy as np  # arrays, math helpers
import pandas as pd  # tables / dataframes

# Visualization libraries
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # nicer plotting styles and helper functions
```
Explanation: Installs and imports the libraries used elsewhere. The `%pip install` line is for interactive notebooks only; the rest bring tools for data, math, and plotting.

---

### Cell: Reproducibility
```python
# Reproducibility
# Set a random seed so results are repeatable each run
np.random.seed(2026) # To ensures the experiments produce consistent and reproducible results across runs.
```
Explanation: Fixes randomness so you get the same train/test split and other pseudo-random behavior every time.

---

### Cell: ucimlrepo install & import
```python
# Install dataset helper library (run once)
%pip install ucimlrepo

# To fetch datasets from the UCI Machine Learning Repository
from ucimlrepo import fetch_ucirepo
```
Explanation: Adds and imports a helper to download the UCI datasets used in the project.

---

### Cell: Fetch Adult Income dataset
```python
# fetch dataset 1 : Adult Income dataset
adult = fetch_ucirepo(id=2)  # download dataset with id 2 from ucimlrepo

# Extract features and target
X = adult.data.features # type: ignore  # feature table
y = adult.data.targets # type: ignore   # target column or series

# Combine features and target into a single DataFrame for easier manipulation
data = pd.concat([X, y], axis=1)

#Print dataframe
display(data.head())  # show first rows
print("Shape:", data.shape)  # show (rows, cols)
display(data.info())  # show column info and dtypes
```
Explanation: Downloads the Adult dataset, puts features and target together, and prints basic info so you can see what the data looks like.

---

### Cell: Cleaning function for Adult dataset
```python
import pandas as pd
import numpy as np

# Cleaning the data

def clean_adult_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Adult Income dataset by standardizing strings,
    handling missing values, fixing label formatting,
    and removing duplicate records.
    """
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Identify all categorical (string) columns
    categorical_cols = df.select_dtypes(include="object").columns

    # Strip leading/trailing whitespace from string columns
    for col in categorical_cols:
        df[col] = df[col].str.strip()

    # Convert Adult dataset's '?' placeholder to proper missing values
    df = df.replace("?", np.nan)

    # Fix income labels in the test set (e.g., '>50K.' -> '>50K')
    if "income" in df.columns:
        df["income"] = df["income"].str.replace(".", "", regex=False)

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    return df
```
Explanation: This function tidies string columns, replaces `?` with real missing values, normalizes income labels, and drops duplicate rows so the data is cleaner for modelling.

---

### Cell: Apply cleaning and save
```python
data = clean_adult_df(data)  # run cleanup
# Check for missing values
display(data.isnull().sum().sort_values(ascending=False))  # show how many missing values per column

# Also print to csv file
data.to_csv('adult_income_dataset.csv', index=False)  # save cleaned dataset for later use
```
Explanation: Cleans the dataset, shows how many missing values remain, and saves a CSV copy.

---

### Cell: Summary statistics
```python
# summary statistics
summary_stats = data.describe()  # numeric summary: mean, std, quartiles
summary_stats
```
Explanation: Gives a quick numeric summary of the dataset (counts, means, etc.). Useful to spot odd values or scale differences.

---

### Cell: Define target and sensitive attribute
```python
# Extract target and sensitive attribute names
if 'income' in data.columns:
    target_col = 'income'
else:
    # fall back to last column if naming differs
    target_col = data.columns[-1]

# Map income to binary target
y = (data[target_col].astype(str).str.strip() == ">50K").astype(int)  # 1 for >50K, else 0
X = data.drop(columns=[target_col])  # features only

# Choose sensitive attribute (prefer 'sex', else try 'race')
for cand in ['sex', 'race']:
    if cand in X.columns:
        sensitive_attr = cand
        break
else:
    # if none of the common names exist, choose the first categorical column
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    sensitive_attr = cat_cols[0] if cat_cols else None

print('Target column:', target_col)
print('Sensitive attribute chosen:', sensitive_attr)
```
Explanation: Builds the model target (0/1) from the income column, keeps features separate, and picks a protected attribute (`sex` or `race`) for fairness analysis.

---

### Cell: Train/test split and remove sensitive attribute from features
```python
# Train/test split with stratification to maintain class balance in both sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2026
)

# Also extract sensitive attribute arrays for later group metrics
A_train = X_train[sensitive_attr].copy() if sensitive_attr is not None else None
A_test = X_test[sensitive_attr].copy() if sensitive_attr is not None else None

# Remove sensitive attribute from features before modeling to prevent direct discrimination
if sensitive_attr is not None:
    X_train = X_train.drop(columns=[sensitive_attr])
    X_test = X_test.drop(columns=[sensitive_attr])

print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)
```
Explanation: Splits the data into training and testing sets while keeping the class distribution the same, saves the sensitive attribute separately for fairness checks, and removes it from the inputs to avoid direct use in the model.

---

### Cell: Preprocessing and baseline logistic regression (Adult)
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Separate numeric and categorical columns
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# Numeric transformer: fill missing values with median, then scale
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical transformer: fill missing with most frequent, then one-hot encode
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Column transformer applies the two transformers to their respective columns
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Full pipeline: preprocessing then logistic regression
baseline_model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# Fit baseline model
baseline_model.fit(X_train, y_train)
```
Explanation: Sets up standard preprocessing for numeric and categorical data, builds a pipeline with logistic regression, and trains the baseline model on the training set.

---

### Cell: Evaluate Accuracy + AUC (Adult)
```python
y_pred = baseline_model.predict(X_test)  # predicted labels
y_prob = baseline_model.predict_proba(X_test)[:, 1]  # predicted probabilities for positive class

print("Accuracy:", accuracy_score(y_test, y_pred))  # fraction correct
print("AUC:", roc_auc_score(y_test, y_prob))  # area under ROC curve
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # precision/recall per class
```
Explanation: Prints simple performance numbers: accuracy (how often the model is right) and AUC (how well it ranks positives vs negatives), plus a full classification report.

---

### Cell: ROC Curve (Adult)
```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)  # false positive and true positive rates for thresholds

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Adult Baseline")
plt.legend()
plt.show()
```
Explanation: Shows the ROC curve. The closer the curve is to the top-left, the better the model. The AUC number in the legend summarizes this as a single value.

---

### Cell: Confusion matrix (Adult)
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)  # 2x2 matrix of counts

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Adult Baseline")
plt.show()
```
Explanation: Visual table of true/false positives/negatives. Useful to see the kinds of mistakes the model makes.

---

### Cell: Demographic Parity - function
```python
def demographic_parity_diff(y_pred, A):
    # Convert inputs to Series with fresh integer index
    A = pd.Series(A).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    rates = {}
    # For each group, compute positive prediction rate
    for group in A.unique():
        rates[group] = y_pred[A == group].mean()

    # Return difference between highest and lowest positive rates and the rates dict
    return max(rates.values()) - min(rates.values()), rates
```
Explanation: Computes how differently the model gives positive predictions across groups; a larger number means more disparity.

---

### Cell: Compute and print Demographic Parity for Adult
```python
# Compute demographic parity difference and per-group positive rates

dp_diff, dp_rates = demographic_parity_diff(y_pred, A_test)

print("Demographic Parity Difference:", dp_diff)
print("Positive prediction rates by group:", dp_rates)
```
Explanation: Shows the DP gap and the per-group rates so you can see which group gets more positive predictions.

---

### Cell: Demographic Parity bar plot (Adult)
```python
dp_rates_series = pd.Series(dp_rates)

plt.figure(figsize=(6,5))
dp_rates_series.plot(kind="bar")
plt.ylabel("Positive Prediction Rate")
plt.title("Demographic Parity - Positive Rates by Sex")
plt.xticks(rotation=0)
plt.show()
```
Explanation: A simple bar chart showing how the model's positive prediction rate varies by sex.

---

### Cell: (Adult) verbal note about gap
```python
# Large gap = more unfair
# In this case, the demographic parity difference is 0.15,
# which indicates that the model is more likely to predict a positive outcome for one group (males) compared to the other group (females).
# This suggests that the model may be exhibiting bias against the group with the lower positive prediction rate,
# and efforts to mitigate this disparity could be considered to improve fairness.
```
Explanation: A plain note interpreting the DP gap: 0.15 means one group gets positive predictions ~15% more often.

---

### Cell: Equal Opportunity - function
```python
def equal_opportunity_diff(y_true, y_pred, A):
    # Convert inputs to Series with fresh index
    A = pd.Series(A).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    tprs = {}
    # For each group, compute true positive rate among actual positives
    for group in A.unique():
        idx = (A == group)
        positives = (y_true[idx] == 1)
        if positives.sum() > 0:
            tprs[group] = (y_pred[idx][positives] == 1).mean()
        else:
            tprs[group] = np.nan

    # Return the difference (max - min) and raw TPRs
    return np.nanmax(list(tprs.values())) - np.nanmin(list(tprs.values())), tprs
```
Explanation: Computes TPR (sensitivity) per group and returns the gap. Smaller is better if fairness is the goal.

---

### Cell: Compute and print Equal Opportunity (Adult)
```python
eo_diff, eo_tprs = equal_opportunity_diff(y_test, y_pred, A_test)

print("Equal Opportunity Difference (TPR Gap):", eo_diff)
print("TPR by group:", eo_tprs)
```
Explanation: Shows how TPR differs between groups. A non-zero gap means the model is better at finding positives for some groups than others.

---

### Cell: Equal Opportunity bar plot (Adult)
```python
eo_tprs_series = pd.Series(eo_tprs)

plt.figure(figsize=(6,5))
eo_tprs_series.plot(kind="bar")
plt.ylabel("True Positive Rate")
plt.title("Equal Opportunity - TPR by Sex")
plt.xticks(rotation=0)
plt.show()
```
Explanation: Bar chart visualizing the TPR per group so you can see whether bands are similar.

---

### Cell: (Adult) verbal note about TPR gap
```python
# Ideally bars should be similar height
# In this case, the equal opportunity difference (TPR gap) is 0.10,
# which indicates that the model has a higher true positive rate for one group (males) compared to the other group (females).
```
Explanation: A 0.10 gap means one group gets true positives about 10 percentage points more often.

---

### Cell: Baseline results table (Adult)
```python
import pandas as pd

results = {
    "dataset": ["Adult Income"],
    "accuracy": [accuracy_score(y_test, y_pred)],
    "auc": [roc_auc_score(y_test, y_prob)],
    "dp_diff": [dp_diff],
    "eo_diff": [eo_diff]
}
results_df = pd.DataFrame(results)
display(results_df)
results_df.to_csv("baseline_results_adult.csv", index=False)
```
Explanation: Collects key metrics into a table and saves them to `baseline_results_adult.csv` so you can compare later.

---

### Cell: Fetch Heart Disease dataset
```python
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# Fetch dataset 2 : Heart Disease Dataset
heart = fetch_ucirepo(id=45)

# Extract features and target
X_hd = heart.data.features # type: ignore
y_hd = heart.data.targets # type: ignore

# Combine into single dataframe
data_hd = pd.concat([X_hd, y_hd], axis=1)

# Display basic info
display(data_hd.head())
print("Shape:", data_hd.shape)
display(data_hd.info())
```
Explanation: Downloads and shows the heart disease dataset, similar to how the Adult data was handled.

---

### Cell: Clean heart function
```python
# Cleaning the Heart Disease dataset

def clean_heart_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    """
    Clean the Heart Disease dataset by standardizing strings,
    handling missing values, and removing duplicates.
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Convert numeric columns where possible (explicitly catch failures)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Drop duplicate rows
    df = df.drop_duplicates()

    return df
```
Explanation: Similar cleaning logic as the Adult dataset, plus attempts to convert numeric-looking columns to numeric dtype.

---

### Cell: Apply heart cleaning and save
```python
data_hd = clean_heart_df(data_hd)
# Check for missing values
display(data_hd.isnull().sum().sort_values(ascending=False))

# Also print to csv file
data_hd.to_csv('heart_disease_dataset.csv', index=False)
```
Explanation: Cleans heart data, shows missing values, and saves a CSV.

---

### Cell: Heart summary stats
```python
# summary statistics
summary_stats = data_hd.describe()
summary_stats
```
Explanation: Numeric summary of the heart dataset for quick checks.

---

### Cell: Define heart target and sensitive attribute
```python
# Identify the target column (usually 'num' in UCI Heart Disease)
if "num" in data_hd.columns:
    target_col_hd = "num"
else:
    target_col_hd = data_hd.columns[-1]

# Convert target to binary: 0 = no disease, 1 = disease (num > 0)
y_hd = (pd.to_numeric(data_hd[target_col_hd], errors="coerce") > 0).astype(int)

# Features
X_hd = data_hd.drop(columns=[target_col_hd])

# Choose aensitive attribute (usually 'sex')
sensitive_attr_hd = "sex" if "sex" in X_hd.columns else None

print("Target column (Heart):", target_col_hd)
print("Sensitive attribute chosen (Heart):", sensitive_attr_hd)
```
Explanation: Sets up the heart dataset target and chooses `sex` as the protected attribute if available.

---

### Cell: Heart train/test split and remove sensitive attribute
```python
# Train/test split with stratification to maintain class balance in both sets
from sklearn.model_selection import train_test_split

X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(
    X_hd, y_hd,
    test_size=0.2,
    stratify=y_hd,
    random_state=2026,
)

# Save sensitive attribute for fairness metrics
A_train_hd = X_train_hd[sensitive_attr_hd].copy() if sensitive_attr_hd else None
A_test_hd  = X_test_hd[sensitive_attr_hd].copy() if sensitive_attr_hd else None

# Remove sensitive attribute from model inputs
if sensitive_attr_hd:
    X_train_hd = X_train_hd.drop(columns=[sensitive_attr_hd])
    X_test_hd  = X_test_hd.drop(columns=[sensitive_attr_hd])

print("Train shape (Heart):", X_train_hd.shape, "Test shape (Heart):", X_test_hd.shape)
```
Explanation: Same pattern as Adult: split, remember the sensitive attribute for fairness checks, and remove it from features.

---

### Cell: Preprocessing and baseline logistic regression (Heart)
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Identify numeric vs categorical columns
numeric_features_hd = X_train_hd.select_dtypes(include=["int64", "float64"]).columns
categorical_features_hd = X_train_hd.select_dtypes(include=["object"]).columns

numeric_transformer_hd = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_hd = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_hd = ColumnTransformer([
    ("num", numeric_transformer_hd, numeric_features_hd),
    ("cat", categorical_transformer_hd, categorical_features_hd)
])

baseline_model_hd = Pipeline([
    ("preprocess", preprocessor_hd),
    ("clf", LogisticRegression(max_iter=2000))
])

baseline_model_hd.fit(X_train_hd, y_train_hd)
```
Explanation: Prepares and trains a logistic regression pipeline for the heart dataset, mirroring the Adult pipeline.

---

### Cell: Evaluate Heart Accuracy + AUC
```python
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

y_pred_hd = baseline_model_hd.predict(X_test_hd)
y_prob_hd = baseline_model_hd.predict_proba(X_test_hd)[:, 1]

print("Heart Accuracy:", accuracy_score(y_test_hd, y_pred_hd))
print("Heart AUC:", roc_auc_score(y_test_hd, y_prob_hd))
print("\nHeart Classification Report:\n", classification_report(y_test_hd, y_pred_hd))
```
Explanation: Prints performance numbers for the heart model (accuracy, AUC, and class-wise metrics).

---

### Cell: Heart ROC Curve
```python
from sklearn.metrics import roc_curve

fpr_hd, tpr_hd, _ = roc_curve(y_test_hd, y_prob_hd)

plt.figure(figsize=(6,5))
plt.plot(fpr_hd, tpr_hd, label=f"AUC = {roc_auc_score(y_test_hd, y_prob_hd):.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Heart Disease Baseline")
plt.legend()
plt.show()
```
Explanation: ROC plot for the heart model; AUC in legend summarizes performance.

---

### Cell: Heart confusion matrix
```python
from sklearn.metrics import confusion_matrix

cm_hd = confusion_matrix(y_test_hd, y_pred_hd)

plt.figure(figsize=(6,5))
sns.heatmap(cm_hd, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Heart Disease Baseline")
plt.show()
```
Explanation: Shows counts of true/false positives/negatives for the heart model.

---

### Cell: Demographic parity (Heart)
```python
# Reuse demographic_parity_diff defined earlier

dp_diff_hd, dp_rates_hd = demographic_parity_diff(y_pred_hd, A_test_hd)

print("Heart Demographic Parity Difference:", dp_diff_hd)
print("Heart Positive prediction rates by group:", dp_rates_hd)
```
Explanation: Computes DP gap for the heart dataset and prints the per-group positive rates.

---

### Cell: Demographic parity plot (Heart)
```python
dp_rates_hd_series = pd.Series(dp_rates_hd)

plt.figure(figsize=(6,5))
dp_rates_hd_series.plot(kind="bar")
plt.ylabel("Positive Prediction Rate")
plt.title("Demographic Parity - Positive Rates by Sex (Heart)")
plt.xticks(rotation=0)
plt.show()
```
Explanation: Bar chart showing positive prediction rates for groups in the heart dataset.

---

### Cell: (Heart) verbal note about DP gap
```python
# Large gap = more unfair
# In this case, the demographic parity difference is 0.20,
# which indicates that the model is more likely to predict a positive outcome for one group (males) compared to the other group (females).
# This suggests that the model may be exhibiting bias against the group with the lower positive prediction rate,
# and efforts to mitigate this disparity could be considered to improve fairness.
```
Explanation: Interprets the DP gap in plain language.

---

### Cell: Equal Opportunity (Heart)
```python
eo_diff_hd, eo_tprs_hd = equal_opportunity_diff(y_test_hd, y_pred_hd, A_test_hd)

print("Heart Equal Opportunity Difference (TPR Gap):", eo_diff_hd)
print("Heart TPR by group:", eo_tprs_hd)
```
Explanation: Computes and prints the TPR gap for the heart dataset.

---

### Cell: Equal Opportunity plot (Heart)
```python
eo_tprs_hd_series = pd.Series(eo_tprs_hd)

plt.figure(figsize=(6,5))
eo_tprs_hd_series.plot(kind="bar")
plt.ylabel("True Positive Rate")
plt.title("Equal Opportunity - TPR by Sex (Heart)")
plt.xticks(rotation=0)
plt.show()
```
Explanation: Shows TPR by group for the heart dataset.

---

### Cell: (Heart) verbal note about TPR gap
```python
# Ideally bars should be similar height
# In this case, the equal opportunity difference (TPR gap) is 0.10,
# which indicates that the model has a higher true positive rate for one group (males) compared to the other
```
Explanation: A short note: a 0.10 TPR gap means some imbalance in true positive detection across groups.

---

### Cell: Heart baseline results table
```python
results_hd = {
    "dataset": ["Heart Disease"],
    "accuracy": [accuracy_score(y_test_hd, y_pred_hd)],
    "auc": [roc_auc_score(y_test_hd, y_prob_hd)],
    "dp_diff": [dp_diff_hd],
    "eo_diff": [eo_diff_hd]
}

results_hd_df = pd.DataFrame(results_hd)
display(results_hd_df)
results_hd_df.to_csv("baseline_results_heart.csv", index=False)
```
Explanation: Collects heart metrics into a table and saves them for comparison.

---

### Cell: Combined results (Adult + Heart)
```python
# Combine Adult + Heart into one table
combined_results_df = pd.concat([results_df, results_hd_df], ignore_index=True)
display(combined_results_df)

combined_results_df.to_csv("baseline_results_combined.csv", index=False)
```
Explanation: Puts both datasets' baseline metrics side-by-side and saves to CSV for convenience.

---

# Quick plain-English summary of all results
- **Accuracy**: shows how often the model got the label right. Higher is better for simple performance.
- **AUC**: shows overall ranking ability; closer to 1 is better.
- **Demographic Parity Difference (dp_diff)**: how differently the model gives positive outcomes across protected groups. Lower is fairer (0 is perfectly equal).
- **Equal Opportunity Difference (eo_diff)**: the gap in true positive rate across groups. Lower is fairer for detecting actual positives equally.

If you want, I can now:
- Insert these commented code blocks directly into the notebook cells, replacing the original code (in-place), or
- Keep this annotated markdown file and optionally commit/save it, or
- Produce a cleaned PDF or share a shorter summary slide with the plain-language results.

Tell me which of the three you prefer and I'll continue.