
# 📊 Sales Forecasting using Time Series Analysis

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('data/sales_data.csv', parse_dates=['Date'], index_col='Date')

# Data Preprocessing
data = data.fillna(method='ffill')

# Plot sales trend
plt.figure(figsize=(10, 6))
plt.plot(data['Sales'], label='Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Trend Over Time')
plt.legend()
plt.show()

# Feature Engineering
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day

# Splitting Data
X = data[['Year', 'Month', 'Day']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales')
plt.plot(y_pred, label='Predicted Sales')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()
