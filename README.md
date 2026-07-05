# Real Estate Price Prediction and Market Segmentation (PACE Framework)

This project applies both Multiple Linear Regression and Logistic Regression models to analyze housing prices based on property characteristics, following the PACE (Plan, Analyze, Construct, Execute) framework taught in the Google Advanced Data Analytics Professional Certificate.

## Key Findings and Business Insights
* Size Impact: Every additional square foot increases the property value by $147.19.
* Location Penalty: For each mile further away from the city center, the price drops by $2,985.75.
* Depreciation: Each year of a house's age decreases its value by $1,107.41.
* Market Segmentation: Properties are classified into "Premium" vs "Standard" tiers using the market price median as the threshold.

## Model Performance and Statistical Validation

### Part 1: Price Prediction (Multiple Linear Regression)
* R-squared ($R^2$): 0.964 — The model explains 96.4% of the variance in housing prices.
* F-statistic: 3292 (p-value < 0.001), indicating high overall model significance.
* Durbin-Watson: 2.002, proving independence of errors (no autocorrelation).
* Residuals: Validated through visual analysis (Q-Q plot for normality and Scatterplot for homoscedasticity) showing errors properly distributed around zero.

### Part 2: Tier Classification (Logistic Regression)
* Objective: Predict the probability of a house qualifying as a "Premium" property based on its characteristics.
* Performance: As documented in image_5548a7.png, the model achieved a stable accuracy of 0.92 (92%) on unseen test data.
* Metrics: Precision, Recall, and F1-Score all reached 0.92 for both classes (Standard and Premium), showing balanced classification performance.
* Optimization: The optimization terminated successfully in 10 iterations with a final function value of 0.142606.

## Technologies Used
* Python 3
* Pandas and NumPy (Data Manipulation and Feature Engineering)
* Matplotlib and Seaborn (Exploratory Data Analysis and Diagnostic Visualizations)
* Statsmodels (OLS Regression, Logit Framework, and Statistical Inference)
* Scikit-Learn (Train/Test Split and Classification Evaluation Metrics)
