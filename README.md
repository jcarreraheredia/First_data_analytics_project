# Data Analytics & Predictive Machine Learning Project

## Project Overview
A brief, 2-3 sentence summary of the business problem or dataset you are analyzing.

## Phase 1: Statistical Baselines (Regression)
- **Objective:** Establish initial benchmarks using statistical modeling.
- **Models Used:** Multiple Linear Regression / Logistic Regression.
- **Key Findings:** What did the regression tell you about feature importance or p-values?

## Phase 2: Upgrading to Machine Learning 🚀
- **Objective:** Improve predictive performance by introducing non-linear machine learning algorithms.
- **Models Evaluated:** Random Forest, XGBoost, etc.
- **Hyperparameter Tuning:** Optimized using `GridSearchCV`.

## Results & Model Comparison
Create a clean Markdown table comparing your baseline regression model against your new ML models.

| Model | Accuracy / $R^2$ | Precision / MAE | ROC-AUC |
| :--- | :--- | :--- | :--- |
| Baseline Logistic Regression | 0.76 | 0.72 | 0.78 |
| Random Forest (Tuned) | **0.87** | **0.84** | **0.91** |

*Summary of results:* "Upgrading to a Random Forest Classifier reduced false positives by X% and improved overall area under the curve (AUC) by Y% compared to the baseline statistical model."

## How to Run the Project
1. Clone the repo...
2. Install dependencies: `pip install -r requirements.txt`
