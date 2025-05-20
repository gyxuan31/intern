import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
airline_data = pd.read_csv(url, index_col='Month', parse_dates=True)

# Plot the dataset
# plt.figure(figsize=(10, 5))
# plt.plot(airline_data)
# plt.title('Monthly Airline Passengers')
# plt.xlabel('Date')
# plt.ylabel('Number of Passengers')
# plt.show()

# Check for stationarity
result = adfuller(airline_data['Passengers'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Since the p-value is > 0.05, the data is not stationary. We need to difference it.
airline_data_diff = airline_data.diff().dropna() # 使用差分使数据变平稳 Yt = Xt - Xt-1

# Check for stationarity again
result = adfuller(airline_data_diff['Passengers'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Plot the differenced data
# plt.figure(figsize=(10, 5))
# plt.plot(airline_data_diff)
# plt.title('Differenced Monthly Airline Passengers')
# plt.xlabel('Date')
# plt.ylabel('Number of Passengers')
# plt.show()

# Fit the ARMA(1, 1) model
model = ARIMA(airline_data_diff, order=(1, 0, 1))  #(p,d,q) order
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Make predictions
start = len(airline_data_diff)
end = start + 20
predictions = model_fit.predict(start=start, end=end)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(airline_data_diff, label='Differenced Original Series')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.title('ARMA Model Predictions on Airline Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()