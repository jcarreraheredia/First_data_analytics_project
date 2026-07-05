"""
Real Estate Price Prediction Model
Author: Jose Carrera
Framework: PACE (Plan, Analyze, Construct, Execute)
Reference: Google Advanced Data Analytics Certification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ==========================================
# 1. PLAN FASE: Data Generation & Setup
# ==========================================
print("--- [PLAN] Generating Dataset ---")

# Seed definition for reproducibility
np.random.seed(42)
n_samples = 500

# Feature Generation
sqft = np.random.normal(2000, 500, n_samples).astype(int)
bedrooms = np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.15, 0.5, 0.25, 0.1])
age = np.random.randint(0, 50, n_samples)
distance_to_center = np.random.uniform(1, 20, n_samples)  # in miles

# Target Generation (Price) with random noise
noise = np.random.normal(0, 15000, n_samples)
price = 50000 + (sqft * 145) + (bedrooms * 12000) - (age * 1100) - (distance_to_center * 2800) + noise

# Construct DataFrame
df = pd.DataFrame({
    'Precio': price.astype(int),
    'Pies_Cuadrados': sqft,
    'Habitaciones': bedrooms,
    'Antiguedad': age,
    'Distancia_Centro': distance_to_center
})

print(f"Dataset generated successfully with shape: {df.shape}\n")

# ==========================================
# 2. ANALYZE FASE: Exploratory Data Analysis
# ==========================================
print("--- [ANALYZE] Running Exploratory Data Analysis ---")

# Heatmap Setup
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación - Características de Vivienda')
plt.tight_layout()

# Pairplot Setup
sns.pairplot(df)

# Display analytical visualizations
plt.show()

# ==========================================
# 3. CONSTRUCT FASE: Model Training & Evaluation
# ==========================================
print("--- [CONSTRUCT] Training Multiple Linear Regression Model ---")

# Define dependent and independent variables
y = df['Precio']
X = df[['Pies_Cuadrados', 'Habitaciones', 'Antiguedad', 'Distancia_Centro']]

# Add constant intercept for statsmodels OLS
X_const = sm.add_constant(X)

# Fit OLS Model
modelo = sm.OLS(y, X_const).fit()

# Display full statistical summary
print(modelo.summary())

# ==========================================
# 4. EXECUTE FASE: Assumption Validation (Residuals)
# ==========================================
print("\n--- [EXECUTE] Validating Regression Assumptions ---")

# Check Residuals Distribution
residuos = modelo.resid

plt.figure(figsize=(8, 6))
sns.histplot(residuos, kde=True)
plt.title('Distribución de Residuos (Normalidad)')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.tight_layout()

# Display assumption check visualization
plt.show()

print("\nProcess finished. Model is fully validated.")