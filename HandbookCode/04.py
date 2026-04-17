# Chapter 4: Granger Causality Test
# this chapter is about figuring out if one time series actually causes another one
# we use something called the granger causality test to check this

# importing all the libraries we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

# --- HELPER FUNCTIONS ---
# these are functions we use over and over again so we define them here

# this function makes lag plots to visually check if the data is stationary
def lag_plots(df):
    fig, axes = plt.subplots(1, len(df.columns), figsize=(len(df.columns)*5, 5))
    for i, col in enumerate(df.columns):
        lag_plot(df[col], lag=1, ax=axes[i])
        axes[i].set_title(col)

# this function runs the kpss test for stationarity
# null hypothesis: the series IS stationary
# if p-value < 0.05, we reject and say its NOT stationary
def kpss_test(df):
    print("KPSS Test Results:")
    for col in df.columns:
        result = kpss(df[col], regression='c', nlags='auto')
        print(f"  {col}: Test Stat = {result[0]:.4f}, p-value = {result[1]:.4f}, "
              f"{'Stationary' if result[1] >= 0.05 else 'NOT Stationary'}")
    print()

# this function runs the adf test for stationarity
# null hypothesis: the series is NOT stationary
# if p-value < 0.05, we reject and say it IS stationary
def adf_test(df):
    print("ADF Test Results:")
    for col in df.columns:
        result = adfuller(df[col], autolag='AIC')
        print(f"  {col}: Test Stat = {result[0]:.4f}, p-value = {result[1]:.4f}, "
              f"{'Stationary' if result[1] < 0.05 else 'NOT Stationary'}")
    print()

# this function splits data into train and test sets (80/20 split)
def splitter(df, train_ratio=0.8):
    n = int(len(df) * train_ratio)
    return df[:n], df[n:]

# this function helps us pick the best lag order p by looking at AIC, BIC, etc.
def select_p(df):
    model = VAR(df)
    results = []
    # test lag orders from 1 to 30
    for i in range(1, 31):
        result = model.fit(i)
        results.append([i, result.aic, result.bic, result.hqic, result.fpe])
    results_df = pd.DataFrame(results, columns=['p', 'AIC', 'BIC', 'HQIC', 'FPE'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    criteria = ['AIC', 'BIC', 'HQIC', 'FPE']
    for ax, crit in zip(axes.flatten(), criteria):
        ax.plot(results_df['p'], results_df[crit], marker='o', markersize=3)
        ax.set_title(crit)
        ax.set_xlabel('Lag Order (p)')
        ax.set_ylabel(crit)
        best_p = results_df.loc[results_df[crit].idxmin(), 'p']
        ax.axvline(x=best_p, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'{crit} (best p={int(best_p)})')
    plt.suptitle('Lag Order Selection')
    plt.tight_layout()

# this function builds a granger causation matrix
# it checks every combination of variables to see which ones granger cause others
# the values are p-values: if p-value < 0.05, then column variable causes row variable
def granger_causation_matrix(data, variables, p, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], p, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(p)]
            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# ============================================================
# EXAMPLE 1: IPO DAM DATASET
# ============================================================
# this dataset has daily measurements of rainfall, oceanic nino index (ONI),
# NIA release flow, and dam water level
print("=" * 60)
print("EXAMPLE 1: IPO DAM DATASET")
print("=" * 60)

ipo_df = pd.read_csv('./data/Ipo_dataset.csv', index_col='Time')
ipo_df = ipo_df.dropna()
print("\nIpo Dam Dataset:")
print(ipo_df.head())

# lets plot all the variables to see what they look like
fig, ax = plt.subplots(4, figsize=(15, 8), sharex=True)
plot_cols = ['Rain', 'ONI', 'NIA', 'Dam']
ipo_df[plot_cols].plot(subplots=True, legend=False, ax=ax)
for a in range(len(ax)):
    ax[a].set_ylabel(plot_cols[a])
ax[-1].set_xlabel('')
plt.suptitle('Ipo Dam Dataset - All Variables')
plt.tight_layout()
plt.show()

# --- Causality between Rainfall and Ipo Dam Water Level ---
print("\n--- Causality between Rainfall and Ipo Dam Water Level ---")

# first we grab just the rain and dam columns
data_df = ipo_df.drop(['ONI', 'NIA'], axis=1)
print(data_df.head())

# checking stationarity with lag plots and tests
print("\nChecking stationarity:")
lag_plots(data_df)
plt.suptitle('Ipo Rain vs Dam - Lag Plots (Before Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: rain is stationary but dam is not

# we need to difference the dam data to make it stationary
print("Differencing Dam data...")
data_df['Dam'] = data_df['Dam'] - data_df['Dam'].shift(1)
data_df = data_df.dropna()

# check stationarity again after differencing
lag_plots(data_df)
plt.suptitle('Ipo Rain vs Dam - Lag Plots (After Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: both are now stationary

# split the data and pick the lag order
train_df, test_df = splitter(data_df)

select_p(train_df)
plt.show()
# we pick p=8

# fit the VAR model and test for granger causality
p = 8
model = VAR(train_df)
var_model = model.fit(p)

print("\nGranger Causation Matrix (Rain vs Dam):")
print(granger_causation_matrix(train_df, train_df.columns, p))
# result: rainfall granger causes dam water level AND dam water level granger causes rainfall
# this is called feedback - they both affect each other
print("\nResult: Rainfall Granger causes Dam water level changes AND vice versa (feedback).")


# --- Causality between NIA Release Flow and Ipo Dam Water Level ---
print("\n--- Causality between NIA Release Flow and Ipo Dam Water Level ---")

data_df = ipo_df.drop(['ONI', 'Rain'], axis=1)
print(data_df.head())

# checking stationarity
print("\nChecking stationarity:")
lag_plots(data_df)
plt.suptitle('Ipo NIA vs Dam - Lag Plots (Before Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: NIA is stationary but dam is not

# differencing
print("Differencing Dam data...")
data_df['Dam'] = data_df['Dam'] - data_df['Dam'].shift(1)
data_df = data_df.dropna()

lag_plots(data_df)
plt.suptitle('Ipo NIA vs Dam - Lag Plots (After Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: both are now stationary

# split and select p
train_df, test_df = splitter(data_df)

select_p(train_df)
plt.show()
# we pick p=8

p = 8
model = VAR(train_df)
var_model = model.fit(p)

print("\nGranger Causation Matrix (NIA vs Dam):")
print(granger_causation_matrix(train_df, train_df.columns, p))
# result: NIA release flow granger causes dam water level AND vice versa (feedback again)
print("\nResult: NIA release flow Granger causes Dam water level changes AND vice versa (feedback).")


# ============================================================
# EXAMPLE 2: LA MESA DAM DATASET
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 2: LA MESA DAM DATASET")
print("=" * 60)

lamesa_df = pd.read_csv('./data/La Mesa_dataset.csv', index_col='Time')
lamesa_df = lamesa_df.dropna()
print("\nLa Mesa Dam Dataset:")
print(lamesa_df.head())

# plot all the variables
fig, ax = plt.subplots(4, figsize=(15, 8), sharex=True)
plot_cols = ['Rain', 'ONI', 'NIA', 'Dam']
lamesa_df[plot_cols].plot(subplots=True, legend=False, ax=ax)
for a in range(len(ax)):
    ax[a].set_ylabel(plot_cols[a])
ax[-1].set_xlabel('')
plt.suptitle('La Mesa Dam Dataset - All Variables')
plt.tight_layout()
plt.show()

# --- Causality between Rainfall and La Mesa Dam Water Level ---
print("\n--- Causality between Rainfall and La Mesa Dam Water Level ---")

data_df = lamesa_df.drop(['ONI', 'NIA'], axis=1)
print(data_df.head())

# check stationarity
print("\nChecking stationarity:")
lag_plots(data_df)
plt.suptitle('La Mesa Rain vs Dam - Lag Plots (Before Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: rain is stationary, dam is not

# differencing
print("Differencing Dam data...")
data_df['Dam'] = data_df['Dam'] - data_df['Dam'].shift(1)
data_df = data_df.dropna()

lag_plots(data_df)
plt.suptitle('La Mesa Rain vs Dam - Lag Plots (After Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: both stationary now

# split and select p
train_df, test_df = splitter(data_df)

select_p(train_df)
plt.show()
# we pick p=7

p = 7
model = VAR(train_df)
var_model = model.fit(p)

print("\nGranger Causation Matrix (Rain vs Dam):")
print(granger_causation_matrix(train_df, train_df.columns, p))
# result: rainfall granger causes dam water level AND vice versa (feedback)
print("\nResult: Rainfall Granger causes La Mesa Dam water level changes AND vice versa (feedback).")


# --- Causality between NIA Release Flow and La Mesa Dam Water Level ---
print("\n--- Causality between NIA Release Flow and La Mesa Dam Water Level ---")

data_df = lamesa_df.drop(['ONI', 'Rain'], axis=1)
print(data_df.head())

# check stationarity
print("\nChecking stationarity:")
lag_plots(data_df)
plt.suptitle('La Mesa NIA vs Dam - Lag Plots (Before Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)

# differencing both since dam is not stationary
print("Differencing Dam data...")
data_df['Dam'] = data_df['Dam'] - data_df['Dam'].shift(1)
data_df = data_df.dropna()

lag_plots(data_df)
plt.suptitle('La Mesa NIA vs Dam - Lag Plots (After Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: both stationary now

# split and select p
train_df, test_df = splitter(data_df)

select_p(train_df)
plt.show()
# we pick p=14

p = 14
model = VAR(train_df)
var_model = model.fit(p)

print("\nGranger Causation Matrix (NIA vs Dam):")
print(granger_causation_matrix(train_df, train_df.columns, p))
# result: NIA release flow and dam water level do NOT granger cause each other for la mesa
# this is different from ipo dam!
print("\nResult: NIA release flow and La Mesa Dam water level do NOT Granger cause each other.")


# ============================================================
# EXAMPLE 3: JENA CLIMATE DATA
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 3: JENA CLIMATE DATA")
print("=" * 60)

# loading the jena climate dataset
# we look at pressure and temperature
train_df = pd.read_csv('./data/train_series_datetime.csv', index_col=0).set_index('Date Time')
val_df = pd.read_csv('./data/val_series_datetime.csv', index_col=0).set_index('Date Time')
test_df = pd.read_csv('./data/test_series_datetime.csv', index_col=0).set_index('Date Time')
train_df.index = pd.to_datetime(train_df.index)
val_df.index = pd.to_datetime(val_df.index)
test_df.index = pd.to_datetime(test_df.index)

train_val_df = pd.concat([train_df, val_df])
jena_df = pd.concat([train_df, val_df, test_df])
print("\nJena Climate Dataset:")
print(jena_df.head())

# --- Causality between Pressure and Temperature ---
print("\n--- Causality between Pressure and Temperature ---")

data_df = jena_df.iloc[:, :2]
print(data_df.head())

# check stationarity
print("\nChecking stationarity:")
lag_plots(data_df)
plt.suptitle('Jena Pressure vs Temperature - Lag Plots (Before Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)

# differencing to make stationary
print("Differencing data...")
for col in data_df.columns:
    data_df[col] = data_df[col] - data_df[col].shift(1)
data_df = data_df.dropna()

lag_plots(data_df)
plt.suptitle('Jena Pressure vs Temperature - Lag Plots (After Differencing)')
plt.tight_layout()
plt.show()

kpss_test(data_df)
adf_test(data_df)
# result: both stationary now

# split and select p
train_df, test_df = splitter(data_df)

select_p(train_df)
plt.show()
# BIC has lowest at p=8, HQIC at p=11, AIC and FPE at p=21 but with elbow
# we pick p=8 for computational efficiency

p = 8
model = VAR(train_df)
var_model = model.fit(p)

print("\nGranger Causation Matrix (Pressure vs Temperature):")
print(granger_causation_matrix(train_df, train_df.columns, p))
# result: pressure granger causes temperature AND temperature granger causes pressure
# another example of feedback
print("\nResult: Pressure Granger causes Temperature AND vice versa (feedback).")

# also testing with higher p=30
print("\nTesting with p=30:")
p = 30
model = VAR(train_df)
var_model = model.fit(p)

print("\nGranger Causation Matrix (Pressure vs Temperature, p=30):")
print(granger_causation_matrix(train_df, train_df.columns, p))
print("\nResult: Same conclusion - Pressure and Temperature Granger cause each other (feedback).")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
We introduced the notion of causality and used the Granger Causality Test
for linear VAR models on several datasets:

1. Ipo Dam: Rain <-> Dam (feedback), NIA <-> Dam (feedback)
2. La Mesa Dam: Rain <-> Dam (feedback), NIA and Dam do NOT cause each other
3. Jena Climate: Pressure <-> Temperature (feedback)

Key takeaway: Granger causality tells us if one variable helps predict another,
but it has limitations - it assumes linear relationships and separable variables.
These limitations are addressed in Chapter 6 with Convergent Cross Mapping.
""")