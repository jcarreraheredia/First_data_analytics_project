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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.20, random_state=42)
# Add constant intercept for statsmodels OLS
X_const = sm.add_constant(X_train)

# Fit OLS Model
modelo = sm.OLS(Y_train, X_const).fit()
x_test_conts= sm.add_constant(X_test)
pred_Model=modelo.predict(x_test_conts)
# Display full statistical summary
print(modelo.summary())

# ==========================================
# 4. EXECUTE FASE: Assumption Validation (Residuals)
# ==========================================
print("\n--- [EXECUTE] Validating Regression Assumptions ---")
#Check R^2 and MAE
print(mean_absolute_error(Y_test,pred_Model))
print(r2_score(Y_test, pred_Model))
# Check Residuals Distribution
residuos = modelo.resid

plt.figure(figsize=(8, 6))
sns.histplot(residuos, kde=True)
plt.title('Distribución de Residuos (Normalidad)')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.tight_layout()

# Create a comprehensive diagnostic layout
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left Plot: Q-Q Plot for Normality
sm.qqplot(residuos, line='s', ax=axes[0])
axes[0].set_title('Gráfico Q-Q (Validación de Normalidad)')

# Right Plot: Residuals vs Fitted Values for Homoscedasticity
sns.scatterplot(x=modelo.fittedvalues, y=residuos, ax=axes[1])
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_title('Residuos vs. Valores Predichos (Homocedasticidad)')
axes[1].set_xlabel('Valores Predichos')
axes[1].set_ylabel('Residuos')

plt.tight_layout()
plt.show()

print("\nProcess finished. Model is fully validated and diagnostic plots generated.")

#Prediction example
casa_nueva=[2300, 3,10,5]
datos_prediccion = [1] + casa_nueva
precio_pred= modelo.predict(datos_prediccion)
print(precio_pred)
# ==========================================
# Real Values vs Predictions
# ==========================================
plt.figure(figsize=(8, 6))

# Real values as blue
plt.scatter(range(len(Y_test)), Y_test, color='blue', alpha=0.6, label='Valores Reales (Datos)', edgecolors='k')

# Predicted values as green
plt.scatter(range(len(pred_Model)), pred_Model, color='green', alpha=0.6, label='Predicciones del Modelo', marker='x')

plt.title('Comparacion de Valores Reales vs. Predicciones (Regresion Lineal)')
plt.xlabel('Indice de la Propiedad en el Set de Prueba')
plt.ylabel('Precio de la Vivienda ($)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ==========================================
# 4. CONSTRUCT & EXECUTE: Model 2 - Logistic Regression
# ==========================================
print("\n--- [CONSTRUCT - MODEL 2] Logistic Regression ---")
df['Premium']=np.where(df['Precio']>df['Precio'].median(),1,0)
y2 = df['Premium']
X2 = df[['Pies_Cuadrados', 'Habitaciones', 'Antiguedad', 'Distancia_Centro']]

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, y2, test_size=0.20, random_state=42)

X_train2_const = sm.add_constant(X_train2)
modelo_logistico = sm.Logit(Y_train2, X_train2_const).fit()

print("\n--- [EXECUTE - MODEL 2] Classification Report ---")
X_test2_const = sm.add_constant(X_test2)
proba = modelo_logistico.predict(X_test2_const)
prediccion_log = np.where(proba > 0.5, 1, 0)

print(classification_report(Y_test2, prediccion_log))
print("\nProcess fully finished. All models successfully validated.")
#Confusion matrix
cm = confusion_matrix(Y_test2, prediccion_log)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Estandar', 'Premium'])
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Matriz de Confusion - Clasificacion de Propiedades')
plt.grid(False)
plt.show()
