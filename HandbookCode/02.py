import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

#from sklearn.model_selection 

# EXAMPLE 1: Univariate LR in Stock Price of Netflix
price = pd.read_csv('./data/NFLX3.csv',usecols=['Date', 'Close'], parse_dates=['Date'])
price.set_index("Date", inplace=True)

plt.rcParams["figure.figsize"] = (15,2)
ax=price.plot()
plt.ylabel('Price')
# plt.grid()
ax.set_title('Daily Close Price')
plt.show()

# Testing for Stationarity
from statsmodels.tsa.stattools import adfuller
series = pd.read_csv('./data/NFLX3.csv',usecols=['Close'])
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# Testing for Autocorrelation
# testing autocorrelation using pandas tools.
plt.rcParams["figure.figsize"] = (6,4)
tsaplots.plot_acf(price)
plt.show()


plt.rcParams["figure.figsize"] = (6,4)
autocorrelation_plot(price)

# Let us create another variable to manipulate rather than using the original data 

window = 30
price2 = price[['Close']]
plt.rcParams["figure.figsize"] = (15,2)
price2.plot()
# plt.grid()

# With the goal of predicting the future prices daily for a month (30days), we create a variable to contain the size
# it will be placed at the end of a variable 'Prediction' in the dataframe that is shifted by 30 days
plt.rcParams["figure.figsize"] = (6,4)
price2['Prediction'] = price2[['Close']].shift(-window)

price2.plot(subplots=True)
plt.show()

print(price2[['Close']].shape)

print(price2[['Prediction']].shape)

# Now we will use the original data on prices as X, the predicted data as y and use 80% of the data as training set 
# while 20% of the data as test set and use sklearn function to train and test the data.
# for both variables, the last 30 values are removed.

X = np.array(price2[['Close']])
X = X[:-window]
print(len(X))

y = np.array(price2['Prediction'])
y = y[:-window]
#print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
#print(X.shape)
lr = LinearRegression()
fit = lr.fit(x_train, y_train)


print(y_train.shape)


## coefficients of linear fit are:
b = fit.intercept_
m = fit.coef_

print("the equation of the linear fit is: ")
print('y= ', m, 'x + ', b)

# checking R^2
R_sqd = lr.score(x_test, y_test)
print("lr confidence: ", R_sqd)


# To show the relationship of the training set and the predicted prices. we plot the following.
plt.rcParams["figure.figsize"] = (6,4)
plt.plot(x_train, y_train, 'o')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.plot(x_train.flatten(), lr.predict(x_train), label='regression line')
plt.legend()
plt.show()


# Here is an attempt to show that the difference between the training set and the predicted values are almost Gaussian
plt.rcParams["figure.figsize"] = (6,4)
plt.hist(lr.predict(x_train)-y_train, bins=np.arange(-125, 126, 25))
plt.show()

# Setting the last 30 rows of the original Close price as test data 
# to predict the next values. 
x_forecast = np.array(price2.drop(['Prediction'], axis=1))[-window:]
print(len(x_forecast))

# # we predict the closing prices...
lr_prediction = lr.predict(x_forecast)
print(len(lr_prediction))

plt.rcParams["figure.figsize"] = (6,4)
plt.plot(x_test, y_test, 'o')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.plot(x_test.flatten(), lr.predict(x_test))
# plt.legend()

plt.figure(figsize=[15, 2])
price2.loc[~price2.Prediction.isnull()].Close.plot(label='y_train')
price2.loc[price2.Prediction.isnull()].Close.plot(label='y_test')
pd.Series(lr.predict(price2.loc[price2.Prediction.isnull()][['Close']].values), 
          index=price2.loc[price2.Prediction.isnull()].index, 
          name='Pred_LR').plot(label='y_pred from regression')
plt.legend(loc=2)
plt.show()

# checking R^2
R_sqd_train = r2_score(y_pred=lr.predict(x_train), y_true=y_train)
print("Train set R^2: ", R_sqd_train)

## R^2 of the predicted vs the actual closing price...
R_sqd_test = r2_score(y_pred=lr_prediction, y_true=y[-window:])
print("Test set R^2: ", R_sqd_test)

# checking for the MAE values:
print("The mean absolute error, MAE is: ", metrics.mean_absolute_error(lr_prediction, x_forecast))

r2_score(price.Close.iloc[:-1], price.Close.iloc[1:])

# Setting the last 30 rows of the original Close price as the variable x_forecast2 and using it as the 
# test data to predict the next values. 

x_forecast2 = np.array(price2.drop(['Prediction'],axis=1))[-window:]
# print(len(x_forecast))

# # we predict the closing prices...
lr_prediction2 = lr.predict(x_forecast2)
print(len(lr_prediction))

import scipy
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_forecast.flatten(), lr_prediction2)

plt.rcParams["figure.figsize"] = (6,4)
plt.plot(x_forecast2.flatten(), lr_prediction2)

# combining the training and validation data sets used in this Jupyter book as our training set in this chapter 
# since I don't need to perform validation.
df= pd.read_csv("./data/train_series.csv")
df2= pd.read_csv("./data/val_series.csv")

## (training + validation series)
df3 = pd.concat([df, df2])

print(df.shape) #train
print(df2.shape) #val
print(df3.shape) #combined

len_train = len(df3)-len(df3)%24
#using the temperature data
temperature = df3[['T (degC)']].iloc[:len_train]
plt.rcParams["figure.figsize"] = (15,2)
ax=temperature.plot()
plt.ylabel('temp')
ax.set_title('Hourly Temp')
plt.show()

window = 24
temperature['Prediction'] = temperature[['T (degC)']].shift(-window)
df4 = pd.read_csv("./data/test_series.csv", usecols=['T (degC)']).iloc[:17520]
test_24 = pd.DataFrame(np.reshape(df4.values, (24, 730)))
# y_test = df4[['T (degC)']].iloc[:24]
X = np.array(temperature.drop(['Prediction'],axis=1))
X = X[:-window]

y = np.array(temperature['Prediction'])
y = y[:-window]
print(X.shape)
print(y.shape)
X.flatten().shape

# Divide training data into 24 hour chunks
# Use 1 Temperature measurement to predict the next 24 hours
X_new = temperature['T (degC)'].iloc[:-window].values.reshape(-1, 24)[:,-1].reshape(-1,1)
y_new = temperature['Prediction'].iloc[:-window].values.reshape(-1, 24)
X_new.shape, y_new.shape

print(X_new)

lr_vectoroutput = LinearRegression()
fit = lr_vectoroutput.fit(X_new, y_new)

X_test = np.vstack([temperature['T (degC)'].iloc[-window:].values.reshape(1,-1),  test_24.iloc[:, :-1].values.T])[:,-1].reshape(-1,1)
y_test = test_24.values.T
## predict for every 24 hrs in test_series
MAE = []
y_pred = lr_vectoroutput.predict(X_test)
for i in range(len(y_test)):
    MAE.append(metrics.mean_absolute_error(y_test[i], y_pred[i]))
print(f"The average Mean Absolute Error is for {len(y_test)} sets of 24-hr data: ", sum(MAE)/730)
# Since the mean absolute error is high, this is not good for LR

# EXAMPLE 3: CARS
#creating dataframe for simple data on cars
cars = pd.read_csv('./data/cars.csv')
cars.head()

from sklearn.linear_model import LinearRegression

# We will model C02 emission (as y variable) using the parameters volume and weight as predictors (x variable)
# for a multi-variate linear regression

params = ['Volume', 'Weight']
X = cars[params]
y = cars['CO2']

# fitting the data with linear regression
lrm = LinearRegression()
model = lrm.fit(X, y)

# coefficients of the fit are as follows:
b = model.intercept_
m = model.coef_

print("Using Multivariate Linear Regression, we have the following equation: ")
print('CO2= ', m[0], 'x1 + ', + m[1], 'x2 + ', b)

#We want to predict CO2 emission given specific weight and volume -- both 1000
model.predict([[1000, 1000]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_pred = lrm.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

# Lasso Regularization... alpha=0 is equivalent to linear regression
from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=0.01)
model3 = lassoreg.fit(X, y)


# coefficients of the fit are as follows:
b = model3.intercept_
m = model3.coef_

print("Using Multivariate Linear Regression, we have the following equation: ")
print('CO2= ', m[0], 'x1 + ', + m[1], 'x2 + ', b)

y_pred = lassoreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

# Ridge Regularization... alpha=0 is equivalent to linear regression
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0.1)
model2 = ridgereg.fit(X, y)

# coefficients of the fit are as follows:
b = model2.intercept_
m = model2.coef_

print("Using Multivariate Linear Regression, we have the following equation: ")
print('CO2= ', m[0], 'x1 + ', + m[1], 'x2 + ', b)

from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_pred = ridgereg.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

# Example 4: Multi-variate Linear Regression on Jena Climate Data
train_df = pd.read_csv('./data/train_series_datetime.csv',index_col=0)
val_df = pd.read_csv('./data/val_series_datetime.csv',index_col=0)
data = pd.concat([train_df, val_df])
data.head()

data.describe()
# print(data.shape)

data_train = data.iloc[:, 1:15]
print(data.shape)

x_train = data_train[['p (mbar)', 'Tpot (K)', 'Tdew (degC)', 
                      'H2OC (mmol/mol)', 'rh (%)', 'VPmax (mbar)', 
                      'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 
                      'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 
                      'wd (deg)']]
y_train = data_train['T (degC)']

data_train.head()

data_train.shape
data_test = pd.read_csv('./data/test_series.csv')
data_test.head()
print(data.shape)
x_test = data_test[['p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'H2OC (mmol/mol)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']]
y_test = data_test[['T (degC)']]
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# Lasso Regularization... alpha=0 is equivalent to linear regression
from sklearn.linear_model import Lasso
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
lassoreg2 = Lasso(alpha=0.01)
model_temp2 = lassoreg2.fit(x_train, y_train)

y_pred3 = lassoreg2.predict(x_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred3))
# Ridge Regularization... alpha=0 is equivalent to linear regression
from sklearn.linear_model import Ridge
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

ridgereg2 = Ridge(alpha=0.01)
model_temp = ridgereg2.fit(x_train, y_train)
y_pred2 = ridgereg2.predict(x_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))
x_train = data_train[['p (mbar)', 'Tpot (K)', 
                      'Tdew (degC)', 'H2OC (mmol/mol)', 
                      'rh (%)', 'VPmax (mbar)', 
                      'VPact (mbar)', 'VPdef (mbar)', 
                      'sh (g/kg)', 'rho (g/m**3)', 
                      'wv (m/s)', 'max. wv (m/s)', 
                      'wd (deg)']].iloc[:len(data_train)-len(data_train)%24]
y_train = data_train['T (degC)'].iloc[:len(data_train)-len(data_train)%24]

X_new = x_train.iloc[::24].values
y_new = y_train.values.reshape(-1, 24)

lr_sim = LinearRegression()
lr_sim.fit(X_new, y_new)

x_test_new = data_test[['p (mbar)', 'Tpot (K)', 
                       'Tdew (degC)', 'H2OC (mmol/mol)', 
                       'rh (%)', 'VPmax (mbar)', 
                       'VPact (mbar)', 'VPdef (mbar)', 
                       'sh (g/kg)', 'rho (g/m**3)', 
                       'wv (m/s)', 'max. wv (m/s)', 
                       'wd (deg)']].iloc[:len(data_test)-len(data_test)%24].iloc[::24].values
y_test_new = data_test[['T (degC)']].iloc[:len(data_test)-len(data_test)%24].values.reshape(-1, 24)

## predict for every 24 hrs in test_series
MAE = []
y_pred = lr_sim.predict(x_test_new)
for i in range(len(y_test_new)):
    MAE.append(metrics.mean_absolute_error(y_test_new[i], y_pred[i]))
print(f"The average Mean Absolute Error is for {len(y_test_new)} sets of 24-hr data: ", sum(MAE)/730)

# Example 5: Extracting the Trend in Climate Data Using MA
# 3420 data is equivalent to 1 month while 83220 is equivalent to two years, 124830 for 3 years
data['MA'] = data['T (degC)'].rolling(window=6840, min_periods=1).mean()
data['EMA'] = data['T (degC)'].ewm(span=3420, adjust=False).mean()

plt.rcParams["figure.figsize"] = (15,2)
data['T (degC)'].plot(label= 'Temp')
data['MA'].plot(label= '6840MA')
data['EMA'].plot(label= '3420EMA')

plt.xlim([0, 83220])
plt.legend(loc='upper left')
plt.show()

# Example 6: Momentum Trading Strategy Using Two MA's
# Recall Netflix data we used above in forecasting using linear regression (LR)
plt.rcParams["figure.figsize"] = (15,2)
price.plot()
plt.ylabel('Price')
plt.show()
## In pandas, the rolling windows is available in computing for the mean of a window sized-n. 
price['20MA'] = price['Close'].rolling(window=20, min_periods=1).mean()
price['65MA'] = price['Close'].rolling(window=65, min_periods=1).mean()

price['Indicator'] = 0.0
price['Indicator'] = np.where(price['20MA']> price['65MA'], 1.0, 0.0)

# To get the difference of the indicators, we use function diff in pandas.
price['Decision'] = price['Indicator'].diff()
# print(price)

#Decision=+1 means buy, while Decision=-1 means sell
# plotting them all together 
price['Close'].plot(label= 'Close')
price['20MA'].plot(label = '20MA')
price['65MA'].plot(label = '65MA')

plt.plot(price[price['Decision'] == 1].index, price['20MA'][price['Decision'] == 1], '^', markersize = 10, color = 'g' , label = 'buy')
plt.plot(price[price['Decision'] == -1].index, price['20MA'][price['Decision'] == -1],  'v', markersize = 10, color = 'r' , label = 'sell')
plt.legend(loc='upper left')
plt.show()

# In pandas, the rolling window equivalent for EMA is 
price['20EMA'] = price['Close'].ewm(span=20, adjust=False).mean()
price['65EMA'] = price['Close'].ewm(span=65, adjust=False).mean()

# Like how we used MA above, we will try to locate the crossover of the two EMAs and create an indicator of crossover 
price['Indicator_EMA'] = 0.0
price['Indicator_EMA'] = np.where(price['20EMA']> price['65EMA'], 1.0, 0.0)

price['Decision_EMA'] = price['Indicator_EMA'].diff()
# print(price)

# Decision=+1 means buy, while Decision=-1 means sell
# plotting them all together 
plt.rcParams["figure.figsize"] = (15,2)
price['Close'].plot(label= 'Close')
price['20EMA'].plot(label = '20EMA')
price['65EMA'].plot(label = '65EMA')

plt.plot(price[price['Decision_EMA'] == 1].index, price['20EMA'][price['Decision_EMA'] == 1], '^', markersize = 10, color = 'g' , label = 'buy')
plt.plot(price[price['Decision_EMA'] == -1].index, price['20EMA'][price['Decision_EMA'] == -1],  'v', markersize = 10, color = 'r' , label = 'sell')
plt.legend(loc='upper left')
plt.show()

# Example 7: Momentum Trading Strategy Using MACD
price['exp1'] = price['Close'].ewm(span=12, adjust=False).mean()
price['exp2'] = price['Close'].ewm(span=26, adjust=False).mean()
price['macd'] = price['exp1']-price['exp2']
price['exp3'] = price['macd'].ewm(span=9, adjust=False).mean()

price['macd'].plot(color = 'g', label = 'MACD')
price['exp3'].plot(color = 'r', label = 'Signal')
plt.legend(loc='upper left')


# Like how we used MA above, we will try to locate the crossover of the MACD and the Signal and plot the indicators 
price['Indicator_MACD'] = 0.0
price['Indicator_MACD'] = np.where(price['macd']> price['exp3'], 1.0, 0.0)

price['Decision_MACD'] = price['Indicator_MACD'].diff()
# print(price)

price['macd'].plot(color = 'g', label = 'MACD')
price['exp3'].plot(color = 'r', label = 'Signal')
plt.legend(loc='upper left')

plt.plot(price[price['Decision_MACD'] == 1].index, price['macd'][price['Decision_MACD'] == 1], '^', markersize = 10, color = 'g' , label = 'buy')
plt.plot(price[price['Decision_MACD'] == -1].index, price['macd'][price['Decision_MACD'] == -1],  'v', markersize = 10, color = 'r' , label = 'sell')
plt.legend(loc='upper left')
plt.show()

price['Close'].plot(color = 'k', label= 'Close')
price['macd'].plot(color = 'g', label = 'MACD')
plt.show()

fig, axs = plt.subplots(2, figsize=(15, 4), sharex=True)
axs[0].set_title('Close Price')
axs[1].set_title('MACD')
axs[0].plot(price['Close'], label='Close')
axs[1].plot(price['macd'], label='macd')
axs[1].set(xlabel='Date')
plt.show()