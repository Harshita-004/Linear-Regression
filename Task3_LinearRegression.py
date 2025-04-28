# Task 3: Linear Regression Solution (Python Script)

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv('dataset/HousingData.csv')  # Ensure correct path

# Data Preprocessing
print("Missing values in each column:\n", df.isnull().sum())
df = df.dropna()

# Exploratory Data Analysis
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Simple Linear Regression (using one feature: 'RM')
X_simple = df[['RM']]
y_simple = df['MEDV']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

# Predictions and Evaluation for Simple Linear Regression
y_pred_s = model_simple.predict(X_test_s)

print("Simple Linear Regression:")
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R2 Score:", r2_score(y_test_s, y_pred_s))

# Plotting Simple Linear Regression
plt.scatter(X_test_s, y_test_s, color='blue')
plt.plot(X_test_s, y_pred_s, color='red')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Price (MEDV)')
plt.title('Simple Linear Regression Line')
plt.show()

# Multiple Linear Regression (using all features)
X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)

# Predictions and Evaluation for Multiple Linear Regression
y_pred_m = model_multiple.predict(X_test)

print("\nMultiple Linear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_m))
print("MSE:", mean_squared_error(y_test, y_pred_m))
print("R2 Score:", r2_score(y_test, y_pred_m))

# Coefficients Interpretation
coeff_df = pd.DataFrame(model_multiple.coef_, X.columns, columns=['Coefficient'])
print("\nCoefficients for Multiple Regression:\n", coeff_df)
