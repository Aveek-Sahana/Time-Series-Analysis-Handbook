#Import required Libraries
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from numpy import log
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from pandas import read_csv
import multiprocessing as mp

### Just to remove warnings to prettify the notebook. 
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./data/wwwusage.csv', names=['value'], header=0)
plt.figure(figsize=(15, 2))
plt.plot(df)
plt.title('Original Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

'''
The null hypothesis of the ADF test is that the time series is non-stationary.
So, if the p-value of the test is less than the significance level (0.05),
then you reject the null hypothesis and infer that the time series is stationary.
'''

plt.rcParams.update({'figure.figsize':(15,8), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

print('ADF Statistic for 1st Order Differencing')
result = adfuller(df.value.diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

print('\n ADF Statistic for 2nd Order Differencing')
result = adfuller(df.value.diff().diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))


# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(15,2.5), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])

plt.show()

# ACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(15,4), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1))
plot_acf(df.value.diff().dropna(), ax=axes[1])

plt.show()

model = ARIMA(df.value, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
# If p values are less than 0.05, it is a significant model
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2, figsize=(15,2.5))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
fig, ax = plt.subplots(figsize=(15, 2))

# Plot original data
df.plot(ax=ax, label='Actual')

# Get predictions
pred = model_fit.get_prediction(start=85, end=100, dynamic=False)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

# Plot predicted mean
pred_mean.plot(ax=ax, label='Forecast')

# Plot confidence interval
ax.fill_between(
    pred_ci.index,
    pred_ci.iloc[:, 0],
    pred_ci.iloc[:, 1],
    alpha=0.3
)

plt.legend()
plt.title("Actual vs Predicted")
plt.show()

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order, train_size=None):
    # prepare training dataset
    X = X.astype('float32')
    if train_size is None:
        train_size = int(len(X) * 0.50)
    else:
        train_size = int(train_size)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        # difference data
        diff = difference(history, 1)
        model = ARIMA(diff, order=arima_order)
        model_fit = model.fit(trend='nc')
        yhat = model_fit.forecast()
        yhat = inverse_difference(history, yhat, 1)
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mae = mean_absolute_error(test, predictions)
    return mae


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s RMSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# USING JENA CLIMATE DATA
train_data = pd.read_csv('./data/train_series.csv',index_col=0).loc[:, ['T (degC)']]
val_data = pd.read_csv('./data/val_series.csv',index_col=0).loc[:, ['T (degC)']]
temp = pd.concat([train_data, val_data]).values  #temperature (in degrees Celsius)
plt.figure(figsize=(15,2))
plt.plot(range(len(temp)), temp)
plt.ylabel('Temperature \n(degree Celcius)')
plt.xlabel('Time (every hour)')
plt.show()
result = adfuller(temp)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# no differencing needed, so d=0
plt.rcParams.update({'figure.figsize':(15,2.5), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2)
axes[0].plot(temp.flatten())
axes[0].set_title('Temperature Signal')
axes[1].set(xlim=(0,20))
plot_pacf(temp.flatten(), ax=axes[1])
plt.show()
# Based on the plot of the PACF, we can see that the function 
# drops to a value of almost zero for lags > 2. 
# So, we can set p to be from 0 to 2. 
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_jena_24hrstep(X, arima_order, train_size=35045):
    # prepare training dataset
    X = X.astype('float32')
    train, test = X[:train_size], X[train_size:]
    test = test[:len(X[train_size:]) - len(X[train_size:])%24]
    test_24 = test.reshape(-1, 24)
    history = train.flatten()

    mae_cv = []
    # There are 730 folds (24 hr chunks), for faster computation, we limit the number of folds to 20 only
    for t in range(len(test_24))[::37]:
        x_cv = np.hstack([history, test_24[:t, :].flatten()])
        y_cv = test_24[t]
        model = ARIMA(x_cv, order=arima_order)
        model_fit = model.fit()
        y_hat = model_fit.forecast(steps=24)
        mae_cv.append(mean_absolute_error(y_cv, y_hat))
    mean_mae = np.mean(mae_cv)
    return mean_mae

selected_order = (1, 0, 2)

def wrapper_fit_arima(x_vals, order=(1, 0, 2)):
    model = ARIMA(x_vals, order=order)
    model_fit = model.fit()
    y_hat = model_fit.forecast(steps=24)
#     print(len(x_vals))
    return y_hat

def evaluate_arima_jena_mp(X, arima_order, train_size=35045):
    # prepare training dataset
    X = X.astype('float32')
    train, test = X[:train_size], X[train_size:]
    test = test[:len(X[train_size:]) - len(X[train_size:])%24]
    test_24 = test.reshape(-1, 24)
    history = train.flatten()
    
    X_cv = []
    Y_cv = []
    for t in range(len(test_24)):
        x_cv = np.hstack([history, test_24[:t, :].flatten()])
        y_cv = test_24[t]
        X_cv.append(x_cv)
        Y_cv.append(y_cv)

    pool = mp.Pool(processes=mp.cpu_count()-4)
    y_hats = pool.map(wrapper_fit_arima, X_cv)        
    
    mae_cv = []
    for t in range(len(test_24)):
        mae_cv.append(mean_absolute_error(Y_cv[t], y_hats[t]))
    mean_mae = np.mean(mae_cv)
    return 

# Load test data
test_data= pd.read_csv('./data/test_series.csv',index_col=0).loc[:, ['T (degC)']]
temp2 = pd.concat([val_data, test_data]).values  #temperature (in degrees Celsius)
## Uncomment code below to fit and predict temperature values in the test set
## By using CPU count of 28, the code ran for about 70 mins. 
# MAE = evaluate_arima_jena_mp(temp2, arima_order=selected_order, train_size=len(val_data))
# print(f'ARIMA MAE: {np.mean(MAE)}') 

## Expected result from running the code above:
## ARIMA MAE: 3.191548582794122
print('ARIMA MAE for Jena test data: 3.191548582794122')
future_data = wrapper_fit_arima(temp2)

plt.figure(figsize=(15,2))
plt.plot(temp2[-500:])
plt.plot(np.arange(24)+500, future_data, label='forecast')
plt.legend()
plt.ylabel('Temperature (degree Celsius)')
plt.show()