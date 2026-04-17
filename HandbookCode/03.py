import numpy as np
import pandas as pd
import statsmodels.tsa as tsa
from statsmodels.tsa.vector_ar.var_model import VAR, FEVD
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, ccf, ccovf, kpss
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import mvts_utils as utils
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Example 1: Just visualizing multivariate
aq_df = pd.read_excel("./data/AirQualityUCI/AirQualityUCI.xlsx", parse_dates=[['Date', 'Time']])\
            .set_index('Date_Time').replace(-200, np.nan).interpolate()
aq_df.head(2)
fig,ax = plt.subplots(5, figsize=(15,8), sharex=True)
plot_cols = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)', 'T', 'RH']
aq_df[plot_cols].plot(subplots=True, legend=False, ax=ax)
for a in range(len(ax)): 
    ax[a].set_ylabel(plot_cols[a])
ax[-1].set_xlabel('')
plt.tight_layout()
#plt.show()

ind_df = pd.read_csv('./data/WorldBankHealth/WorldBankHealthPopulation_SeriesSummary.csv')\
            .loc[:,['series_code', 'indicator_name']].drop_duplicates().reindex()\
            .sort_values('indicator_name').set_index('series_code')
hn_df = pd.read_csv('./data/WorldBankHealth/WorldBankHealthPopulation_HealthNutritionPopulation.csv')\
            .pivot(index='year', columns='indicator_code', values='value')
cols = ['SH.XPD.KHEX.GD.ZS', 'SH.XPD.CHEX.GD.ZS', 'SH.XPD.GHED.GD.ZS']
health_expenditure_df = hn_df.loc[np.arange(2000, 2018), cols]\
    .rename(columns = dict(ind_df.loc[cols].indicator_name\
                           .apply(lambda x: '_'.join(x.split('(')[0].split(' ')[:-1]))))
health_expenditure_df.index = pd.date_range('2000-1-1', periods=len(health_expenditure_df), freq="A-DEC")
health_expenditure_df.head(3)
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15,7))
health_expenditure_df.plot(subplots=True, ax=ax, legend=False)
y_label = ['Capital health expenditure', 'Current health expenditure',
           'Domestic general government\n health expenditure']
for a in range(len(ax)): 
    ax[a].set_ylabel(f"{y_label[a]}\n (% of GDP)")
plt.tight_layout()
#plt.show()

treas_df = pd.read_excel("./data/USTreasuryRates/us-treasury-rates-weekly.xlsx")
treas_df = treas_df.rename(columns={'Unnamed: 0': 'Date'}).set_index('Date')
treas_df.index = pd.to_datetime(treas_df.index)
fig,ax = plt.subplots(1, figsize=(15, 3), sharex=True)
data_df = treas_df.iloc[:, 0:9]
data_df.plot(ax=ax)
plt.ylabel('US Treasury yield (%)')
plt.xlabel('')
plt.legend(ncol=2)
plt.tight_layout()
#plt.show()

jena_data = pd.read_csv('./data/jena_climate_2009_2016.csv')
jena_data['Date Time'] = pd.to_datetime(jena_data['Date Time'], dayfirst=True)
jena_data = jena_data.set_index('Date Time')
jena_data.head(3)
fig,ax = plt.subplots(jena_data.shape[-1], figsize=(15,16), sharex=True)
jena_data.iloc[-1000:].plot(subplots=True, ax=ax)
ax[-1].set_xlabel('')
plt.tight_layout()
plt.show()


# Example 2: Forecasting Air Quality Data (CO, NO2 and RH)
cols = ['CO(GT)', 'NO2(GT)', 'RH']
data_df = aq_df.loc[aq_df.index>'2004-10-01',cols]
fig,ax = plt.subplots(3, figsize=(15,5), sharex=True)
data_df.plot(ax=ax, subplots=True)
plt.xlabel('')
plt.tight_layout()
plt.show()

test_stat, p_val = [], []
cv_1pct, cv_5pct, cv_10pct = [], [], []
for c in data_df.columns: 
    adf_res = adfuller(data_df[c].dropna())
    test_stat.append(adf_res[0])
    p_val.append(adf_res[1])
    cv_1pct.append(adf_res[4]['1%'])
    cv_5pct.append(adf_res[4]['5%'])
    cv_10pct.append(adf_res[4]['10%'])
adf_res_df = pd.DataFrame({'Test statistic': test_stat, 
                           'p-value': p_val, 
                           'Critical value - 1%': cv_1pct,
                           'Critical value - 5%': cv_5pct,
                           'Critical value - 10%': cv_10pct}, 
                         index=data_df.columns).T
adf_res_df.round(4)
print(adf_res_df)

forecast_length = 24 
train_df, test_df = data_df.iloc[:-forecast_length], data_df.iloc[-forecast_length:]
test_df = test_df.filter(test_df.columns[~test_df.columns.str.contains('-d')])
train_df.reset_index().to_csv('./data/AirQualityUCI/train_data.csv')
test_df.reset_index().to_csv('./data/AirQualityUCI/test_data.csv')
fig,ax = plt.subplots(3, figsize=(15, 5), sharex=True)
train_df.plot(ax=ax, subplots=True)
plt.xlabel('')
plt.tight_layout()
plt.show()

aic, bic, fpe, hqic = [], [], [], []
model = VAR(train_df) 
p = np.arange(1,60)
for i in p:
    result = model.fit(i)
    aic.append(result.aic)
    bic.append(result.bic)
    fpe.append(result.fpe)
    hqic.append(result.hqic)
lags_metrics_df = pd.DataFrame({'AIC': aic, 
                                'BIC': bic, 
                                'HQIC': hqic,
                                'FPE': fpe}, 
                               index=p)    
fig, ax = plt.subplots(1, 4, figsize=(15, 3), sharex=True)
lags_metrics_df.plot(subplots=True, ax=ax, marker='o')
plt.tight_layout()
plt.show()

var_model = model.fit(26)
var_model.summary()

forecast_var = pd.DataFrame(var_model.forecast(train_df.values, 
                                              steps=forecast_length), 
                           columns=train_df.columns, 
                           index=test_df.index)
forecast_var = forecast_var.rename(columns={c: c+'-VAR' for c in forecast_var.columns})

for c in train_df.columns:
    fig, ax = plt.subplots(figsize=[15, 2])
    pd.concat([train_df[[c]], forecast_var[[c+'-VAR']]], axis=1).plot(ax=ax)
    plt.xlim(left=pd.to_datetime('2005-03-01'))
    plt.xlabel('')
#     plt.tight_layout()
plt.show()

# For model order selection, refer to Chapter 1
selected_order = {'CO(GT)': [(0, 1, 0)],
                  'NO2(GT)': [(0, 1, 0)],
                  'RH': [(3, 1, 1)]}
forecast_arima = {}
for c in cols:
    forecast_arima[c+'-ARIMA'] = utils.forecast_arima(train_df[c].values, 
                                                      test_df[c].values, 
                                                      order=selected_order[c][0])
forecast_arima = pd.DataFrame(forecast_arima, index=forecast_var.index)
forecast_arima.head()
forecasts = pd.concat([forecast_arima, forecast_var], axis=1)
for c in cols:
    fig, ax = utils.plot_forecasts_static(train_df=train_df,
                                          test_df=test_df, 
                                          forecast_df=forecasts, 
                                          column_name=c,
                                          min_train_date='2005-04-01', 
                                          suffix=['-VAR', '-ARIMA'],
                                          title=c)
pd.concat([utils.test_performance_metrics(test_df, forecast_var, suffix='-VAR'),
           utils.test_performance_metrics(test_df, forecast_arima, suffix='-ARIMA')], axis=1)
irf = var_model.irf(periods=8)
ax = irf.plot(orth=True, 
              subplot_params={'fontsize': 10})
fevd = var_model.fevd(8)
ax = fevd.plot(figsize=(10, 8))
plt.show()

# Example 3: Forecasting the Jena Climate 
train_df = pd.read_csv('./data/train_series_datetime.csv',index_col=0).set_index('Date Time')
val_df = pd.read_csv('./data/val_series_datetime.csv',index_col=0).set_index('Date Time')
test_df = pd.read_csv('./data/test_series_datetime.csv',index_col=0).set_index('Date Time')
train_df.index = pd.to_datetime(train_df.index, dayfirst=True)
val_df.index = pd.to_datetime(val_df.index, dayfirst=True)
test_df.index = pd.to_datetime(test_df.index, dayfirst=True)
train_val_df = pd.concat([train_df, val_df])

test_stat, p_val = [], []
cv_1pct, cv_5pct, cv_10pct = [], [], []
for c in train_val_df.columns: 
    adf_res = adfuller(train_val_df[c].dropna())
    test_stat.append(adf_res[0])
    p_val.append(adf_res[1])
    cv_1pct.append(adf_res[4]['1%'])
    cv_5pct.append(adf_res[4]['5%'])
    cv_10pct.append(adf_res[4]['10%'])
adf_res_df = pd.DataFrame({'Test statistic': test_stat, 
                           'p-value': p_val, 
                           'Critical value - 1%': cv_1pct,
                           'Critical value - 5%': cv_5pct,
                           'Critical value - 10%': cv_10pct}, 
                         index=train_df.columns).T
adf_res_df.round(4)
((adf_res_df.loc['Test statistic']< adf_res_df.loc['Critical value - 1%']) & 
(adf_res_df.loc['Test statistic']< adf_res_df.loc['Critical value - 5%']) &
( adf_res_df.loc['Test statistic']< adf_res_df.loc['Critical value - 10%']))
print(adf_res_df)

aic, bic, fpe, hqic = [], [], [], []
model = VAR(train_val_df)
p = np.arange(1,60)
for i in p:
    result = model.fit(i)
    aic.append(result.aic)
    bic.append(result.bic)
    fpe.append(result.fpe)
    hqic.append(result.hqic)
lags_metrics_df = pd.DataFrame({'AIC': aic, 
                                'BIC': bic, 
                                'HQIC': hqic,
                                'FPE': fpe}, 
                               index=p)    
fig, ax = plt.subplots(1, 4, figsize=(15, 3), sharex=True)
lags_metrics_df.plot(subplots=True, ax=ax, marker='o')
plt.tight_layout()
lags_metrics_df[25:].idxmin()

var_model = model.fit(26)
test_index = np.arange(len(test_df)- (len(test_df))%24).reshape((-1, 24))
fit_index = [test_index[:i].flatten() for i in range(len(test_index))]
forecasts_df = []
for n in range(len(fit_index)):
    forecast_var = pd.DataFrame(var_model.forecast(
        pd.concat([train_val_df, test_df.iloc[fit_index[n]]]).values, steps=24), 
                                columns=train_val_df.columns, 
                                index=test_df.iloc[test_index[n]].index)
    forecasts_df.append(forecast_var)
forecasts_df = pd.concat(forecasts_df)
forecasts_df.columns = forecasts_df.columns+['-VAR']
for c in train_val_df.columns:
    fig, ax = plt.subplots(figsize=[14, 2])
    train_val_df[c].plot(ax=ax)
    forecasts_df[[c+'-VAR']].plot(ax=ax)
    plt.ylim(train_val_df[c].min(), train_val_df[c].max())
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()

utils.test_performance_metrics(test_df.loc[forecasts_df.index], forecasts_df, suffix='-VAR').loc[['MAE', 'MSE']]
T_MAEs = []
for n in range(len(test_index)):
    index = forecasts_df.iloc[test_index[n]].index
    T_MAEs.append(utils.mean_absolute_error(test_df.loc[index, 'T (degC)'].values, 
                                            forecasts_df.loc[index, 'T (degC)-VAR'].values))
print(f'VAR(26) MAE: {np.mean(T_MAEs)}')

all_data = pd.concat([train_df, val_df, test_df])
future_index = [all_data.index[-1]+pd.Timedelta(f'{h} hour') for h in np.arange(24)+1]
forecast_var_future = pd.DataFrame(var_model.forecast(all_data.values, 
                                                      steps=24), 
                                   columns=all_data.columns, 
                                   index=future_index)

forecast_var_future.columns = forecast_var_future.columns+['-VAR']

for c in train_val_df.columns:
    fig, ax = plt.subplots(figsize=[14, 2])
    all_data.iloc[-500:][c].plot(ax=ax)
    forecast_var_future[[c+'-VAR']].plot(ax=ax)
    plt.ylim(train_val_df[c].min(), train_val_df[c].max())
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()