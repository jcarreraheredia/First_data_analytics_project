# ==============================================================================
# PROJECT: PEER-TO-PEER LOAN DEFAULT RISK PREDICTION
# This script performs end-to-end data processing, exploratory data analysis (EDA), 
# anomaly cleaning, and evaluates three different predictive models (Logistic 
# Regression vs. Random Forest vs. Gradient Boosting).
# ==============================================================================

# Import core data manipulation and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning modeling and evaluation tools from Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ------------------------------------------------------------------------------
# STEP 1: LOAD THE DATASET & RUN INITIAL DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------------
# Load the raw peer-to-peer loan data
df = pd.read_csv('loan_data.csv')

# Print the first 5 records to inspect columns and data layout
print(df.head())

# Print summary of features, data types, and non-null values
print(df.info())

# Print basic statistical summaries (min, max, mean, quartiles) for numeric columns
print(df.describe())

# Check for missing (null) values in each column
print(df.isnull().sum())

# ------------------------------------------------------------------------------
# STEP 2: DATA CLEANING & ANOMALY HANDLING (MEDIAN IMPUTATION & OUTLIERS)
# ------------------------------------------------------------------------------
# Address missing values in Credit_Score using the median (robust to skew)
median_value = df['Credit_Score'].median()
df['Credit_Score'] = df['Credit_Score'].fillna(median_value)

# Visualize the distribution of Employment_Duration to spot outlier trends
sns.boxplot(x=df['Employment_Duration'])
plt.show()

# Replace the unrealistic 99.0 years data-entry outlier with the US average (3.9 years)
df['Employment_Duration'] = df['Employment_Duration'].replace(99.0, 3.9)

# ------------------------------------------------------------------------------
# STEP 3: CORRELATION ANALYSIS (CHECKING FOR MULTICOLLINEARITY)
# ------------------------------------------------------------------------------
# Isolate numeric features to generate a Pearson correlation matrix
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

# Plot a correlation heatmap to examine relationships between continuous features
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# ------------------------------------------------------------------------------
# STEP 4: PREPROCESSING & FEATURE ENGINEERING (DUMMY ENCODING & TARGET SPLITTING)
# ------------------------------------------------------------------------------
# Drop the unique borrower identifier column as it has no predictive power
df_model = df.drop('Borrower_ID', axis=1)

# Apply One-Hot Encoding to categorical column 'Loan_Purpose', dropping the first class to avoid dummy trap
df_model = pd.get_dummies(df_model, columns=['Loan_Purpose'], drop_first=True)

# Separate independent variables (features) and the target variable (Default)
X = df_model.drop('Default', axis=1)
y = df_model['Default']

# ------------------------------------------------------------------------------
# STEP 5: TRAIN-TEST SPLIT (WITH CLASS STRATIFICATION)
# ------------------------------------------------------------------------------
# Split into 70% train and 30% test sets. Stratification ensures balanced default ratios in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# ------------------------------------------------------------------------------
# STEP 6: MODEL 1 - BASELINE STATISTICAL MODEL (LOGISTIC REGRESSION)
# ------------------------------------------------------------------------------
# Initialize Logistic Regression with enough iterations to ensure optimization convergence
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict target classes for test features
y_pred = log_reg.predict(X_test)

# Print classification evaluation metrics for our baseline model
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------------------
# STEP 7: ENSEMBLE MODELS - BAGGING (RANDOM FOREST) & BOOSTING (GRADIENT BOOSTING)
# ------------------------------------------------------------------------------
# 1. Bagging Framework: Build a Random Forest Classifier of 100 parallel trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 2. Boosting Framework: Build a sequential Gradient Boosting Classifier of 100 trees
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

# Print comparative classification reports for both ensemble methods
print("\n--- Classification Report for Random forest ---")
print(classification_report(y_test, rf_preds))

print("\n--- Classification Report for Gradient Boosting ---")
print(classification_report(y_test, gb_preds))

# ------------------------------------------------------------------------------
# STEP 8: MODEL INTERPRETABILITY (FEATURE IMPORTANCES FROM RANDOM FOREST)
# ------------------------------------------------------------------------------
# Extract feature importance weights calculated during Random Forest training
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Generate a bar plot to visualize which attributes are driving the model's predictions
plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Random Forest Model")
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.xlabel("Relative Importance")
plt.show()
# Disclaimer: All the comments were generated with AI but the code was completely generated by me