# Real Estate Price Prediction using Multiple Linear Regression (PACE Framework)

This project applies a **Multiple Linear Regression** model to predict housing prices based on property characteristics, following the **PACE** (Plan, Analyze, Construct, Execute) framework taught in the Google Advanced Data Analytics Professional Certificate.

## 📊 Key Findings & Business Insights
* **Size Impact:** Every additional square foot increases the property value by **$147.19**.
* **Location Penalty:** For each mile further away from the city center, the price drops by **$2,985.75**.
* **Depreciation:** Each year of a house's age decreases its value by **$1,107.41**.

## 🛠️ Model Performance & Statistical Validation
* **R-squared ($R^2$):** **0.964** — The model explains 96.4% of the variance in housing prices.
* **F-statistic:** **3292** (p-value < 0.001), indicating high overall model significance.
* **Durbin-Watson:** **2.002**, proving independence of errors (no autocorrelation).
* **Residuals:** Validated through visual analysis (histogram with KDE) showing a normal distribution centered at zero.

## 🚀 Technologies Used
* Python 3
* Pandas & NumPy (Data Manipulation)
* Matplotlib & Seaborn (Exploratory Data Analysis)
* Statsmodels (OLS Regression & Statistical Inference)