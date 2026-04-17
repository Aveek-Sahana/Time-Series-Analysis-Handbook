import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import date, timedelta
import scipy as scp
import random
from statsmodels.tsa.stattools import acf, adfuller, ccf, ccovf
from itertools import repeat
import pywt # Python Wavelets
import os
import warnings
warnings.filterwarnings('ignore')

'''
CROSS CORRELATION
Phenomenon captured by time series may not happen at the exact same time with some lag h between them. 
A simple example of this is Sonar technology. 
The time series associated with the response from the sound waves being reflected comes at some lag 
compared to the time series of the device emitting the initial sound waves. 
It is this lag we want to measure when we use the cross-correlation function (CCF).
'''

# EXAMPLE 1: Temperature and Wind
df = pd.read_csv('./data/jena_climate_2009_2016.csv')
print(df.head())
data = df.iloc[:, 1:].astype(float).to_numpy()
temp = data[:, 1][::144]  # Temperature (in degrees Celsius) at one sample/day
wind = data[:, 10][::144]  # Wind Speed (in m/s) at one sample/day
dur = len(temp)
samp_rate = 1

plt.subplots(figsize=(15, 2))
plt.plot(range(len(temp)), temp, label='Temperature')
plt.plot(range(len(wind)), wind, label='Wind Velocity')
plt.legend()

plt.subplots(figsize=(15, 2))
plt.plot(range(len(temp)), (temp-np.mean(temp)) /
         np.std(temp), label='Temperature')
plt.plot(range(len(wind)), (wind-np.mean(wind)) /
         np.std(wind), label='Wind Velocity')
plt.ylabel('Value')
plt.xlabel('Time (in days)')
plt.legend()

plt.subplots(figsize=(15, 2))

ccf_12 = ccf(temp, wind, unbiased=False)
plt.ylabel('Correlation Coefficient')
plt.xlabel('Lag $h$')
plt.stem(np.linspace(0, dur, samp_rate*dur), ccf_12)
peak = np.argmax(ccf_12)/samp_rate
plt.axvline(peak, c='r')
print('Maximum at:', peak, '(in $s$)')

plt.show()


# EXAMPLE 2: Rain and Flow
flow_df = pd.read_csv('./data/cc/flowsud_2000.txt', sep='\t')
flow_df.columns = ['USGS', 'site_no', 'datetime', 'flow', '-']

flow = flow_df['flow']

rain_df = pd.read_csv('./data/cc/weather_2000.txt')
rain_df['Prcp'] = rain_df['Prcp'].replace('T', 0.0005).astype(float)

rain = rain_df['Prcp']

dur = len(rain)
samp_rate = 1

plt.subplots(figsize=(15, 2))
plt.plot(flow, label='Stream Flow')
plt.plot(rain, label='Precipitation')
plt.legend()

plt.subplots(figsize=(15, 2))
plt.plot((rain-np.mean(rain))/np.std(rain), label='Preciptation (exaggerated)')
plt.plot((flow-np.mean(flow))/np.std(flow), label='Stream Flow')
plt.xlabel('Day of the Year')
plt.legend()

result = adfuller(flow)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))
result = adfuller(rain)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

plt.subplots(figsize=(15, 2))

ccf_12 = ccf(flow, rain, unbiased=False)
plt.ylabel('Correlation Coefficient')
plt.xlabel('Lag $h$')
plt.stem(np.linspace(0, dur, samp_rate*dur), ccf_12)
peak = np.argmax(ccf_12)/samp_rate
plt.axvline(peak, c='r')
print('Maximum at:', peak, '(in $days$)')

plt.show()


'''
Fourier Transform
Converting from time on the x-axis to frequency on the x-axis
'''
def sine_wave(amp=1, freq=1, sample_rate=200, duration=5,
              plot=True, shift=0, noise=0):
    x = np.linspace(0, duration, sample_rate*duration)
    frequencies = x * freq
    y = amp*np.sin((2 * np.pi * frequencies) + shift) + noise
    if plot:
        plt.subplots(figsize=(15, 2))
        plt.plot(x, y)
        plt.show()
    return x, y

# Mixed frequency signal example
samp_rate = 500  # Sampling Frequency in Hertz
dur = 5  # Duration in Seconds

amp1 = 1 # Amplitude of Sine Wave 1
freq1 = 5  # Frequency of Sine Wave 1

amp2 = 1  # Amplitude of Sine Wave 2
freq2 = 17  # Frequency of Sine Wave 2

x1, y1 = sine_wave(amp1, freq1, samp_rate, dur, plot=False)
x2, y2 = sine_wave(amp2, freq2, samp_rate, dur, plot=False)
y = y1 + y2  # Simply overlapping the two signals
plt.subplots(figsize=(15, 2))
plt.xlabel('Time')
plt.plot(x1, y)

y_ft = scp.fft.fft(y)
freqs = scp.fft.fftfreq(x1.shape[-1])*samp_rate

plt.subplots(figsize=(5, 5))
plt.plot(freqs, y_ft.real**2)
plt.xlabel('Frequency (in Hz)')
peaks = scp.signal.find_peaks(y_ft.real**2)
peak_vals = peaks[0][:len(peaks[0])//2]/dur
print('Peaks at:', peak_vals)

plt.show()


# Filtering by energy level to discern signal from noise!
noise = np.random.uniform(-1,1, len(y))

y_noisy = y+noise
plt.subplots(figsize=(15, 2))
plt.plot(x1, y_noisy)
plt.title('Noisy Signal')

y_ft = scp.fft.fft(y_noisy)
freqs = scp.fft.fftfreq(x1.shape[-1])*samp_rate
peaks = scp.signal.find_peaks(y_ft.real**2)
peak_vals = peaks[0][:len(peaks[0])//2]/dur

plt.subplots(figsize=(5, 5))
plt.plot(freqs, y_ft.real**2)
plt.ylabel('Energy')
plt.xlabel('Frequency')

threshold = 3*np.std(y_ft.real)
peaks = scp.signal.find_peaks(y_ft.real, threshold=threshold)
peak_vals = peaks[0][:len(peaks[0])//2]/dur
print('Peaks at:', peak_vals)

y_ft_denoised = [0 if x < threshold else x for x in y_ft]
plt.subplots(figsize=(15, 2))
y_denoised = scp.fft.ifft(y_ft_denoised)
plt.plot(x1, y_denoised.real)
plt.title('Noise Taken out of Signal')

plt.show()

# Using Jena Climate Data - Real
df = pd.read_csv('./data/jena_climate_2009_2016.csv')
df['Date Time'] = pd.to_datetime(
    df['Date Time'], dayfirst=True, format="%d.%m.%Y %H:%M:%S")
df['Date'] = df['Date Time'].dt.date
mean_temps = df.groupby('Date')['T (degC)'].mean().values
plt.subplots(figsize=(15, 2))
plt.plot(temp)
plt.title('Jena Climate Temperature')

y_ft = scp.fft.fft(temp)
y_ft_copy = y_ft.copy()
freqs = scp.fft.fftfreq(len(temp), d=1/365)
y_ft_copy[(abs(freqs) >= 1)] = 0
plt.plot(freqs, abs(y_ft_copy))

y_hf = scp.fft.ifft(y_ft_copy)
plt.subplots(figsize=(15, 2))
# plt.title('Removing Yearly Cycle')
plt.plot(range(len(temp)), y_hf.real)
plt.title('Filtered Jena Climate Temperature')

plt.show()

'''
Wavelet Transform
A wavelet transform (WT) allows you to measure how "much"of a certain type of wavelet there exists
in a given signal. While both FT and WT both transform a time series into the frequency domain, 
a key difference is that WT gives you information about the time domain as well.
'''
def rescale(arr, scale=2):
  n = len(arr)
  return np.interp(np.linspace(0, n, scale*n+1), np.arange(n), arr)
fig, ax = plt.subplots(figsize=(15, 2))
wav1 = pywt.ContinuousWavelet('gaus1')
int_psi1, x = pywt.integrate_wavelet(wav1)
for s in range(5):
  ax.plot(rescale(int_psi1, scale=s), label='Scale: '+ str(s))
  ax.legend()  
  ax.set_title('Gaussian Mother Wavelet')

s1 = list(scp.signal.windows.gaussian(250, std=20))
s1 = np.tile(s1, 4)
plt.subplots(figsize=(15, 2))
plt.xlim(left=0, right=1000)
plt.ylabel('Value')
plt.xlabel('Time (t)')
plt.plot(s1)
plt.show()

coeffs, freqs = pywt.cwt(s1, range(1,250), wavelet='gaus1')
plt.subplots(figsize=(18.75, 2))
plt.xlim(left=0, right=1000)
plt.ylabel('Scale (s)')
plt.xlabel('Time (t)')
plt.imshow(coeffs, aspect='auto')
plt.colorbar()
plt.show()

# Morlet Wavelet
fig, ax = plt.subplots(figsize=(15, 2))
wav2 = pywt.ContinuousWavelet('morl')
int_psi2, x = pywt.integrate_wavelet(wav2)
for s in range(5):
  ax.plot(rescale(int_psi2, scale=s), label='Scale: '+ str(s))
  ax.set_title('Morlet Mother Wavelet')
  ax.legend()
sine5 = sine_wave(freq=5, duration =5, plot=False)[1]
sine10 = sine_wave(freq=10, plot=False)[1]
sine_5_10 = sine5+sine10
plt.subplots(figsize=(15, 2))
plt.plot(sine_5_10)
plt.subplots(figsize=(18.75, 2))
plt.xlim(left=0, right=1000)
plt.ylabel('Scale (s)')
plt.xlabel('Time (t)')
plt.imshow(coeffs, aspect='auto')
plt.colorbar()
plt.show()

# Accelerometer and Gyroscope Data
def load_y_data(y_path):
    y = np.loadtxt(y_path, dtype=np.int32).reshape(-1, 1)
    # change labels range from 1-6 t 0-5, this enables a sparse_categorical_crossentropy loss function
    return y - 1


def load_X_data(X_path):
    X_signal_paths = [X_path + file for file in os.listdir(X_path)]
    X_signals = [np.loadtxt(path, dtype=np.float32) for path in X_signal_paths]
    return np.transpose(np.array(X_signals), (1, 2, 0))


PATH = './data/cwt/'
LABEL_NAMES = ["Walking", "Walking upstairs",
               "Walking downstairs", "Sitting", "Standing", "Laying"]

# load X data
X_train = load_X_data(PATH + 'train/Inertial Signals/')
X_test = load_X_data(PATH + 'test/Inertial Signals/')
# load y label
y_train = load_y_data(PATH + 'train/y_train.txt')
y_test = load_y_data(PATH + 'test/y_test.txt')

fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
pd.DataFrame(X_train[79]).plot(ax=ax[0], title='Walking')
pd.DataFrame(X_train[51]).plot(ax=ax[1], title='Laying')

def split_indices_per_label(y):
    indicies_per_label = [[] for x in range(0, 6)]
    # loop over the six labels
    for i in range(6):
        indicies_per_label[i] = np.where(y == i)[0]
    return indicies_per_label


def plot_cwt_coeffs_per_label(X, label_indicies, label_names, signal, sample, scales, wavelet):

    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True,
                            sharey=True, figsize=(12, 5))

    for ax, indices, name in zip(axs.flat, label_indicies, label_names):
        coeffs, freqs = pywt.cwt(
            X[indices[sample], :, signal], scales, wavelet=wavelet)
        ax.imshow(coeffs, cmap='coolwarm', aspect='auto')
        ax.set_title(name)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Scale')
        ax.set_xlabel('Time')
    plt.tight_layout()


train_labels_indicies = split_indices_per_label(y_train)

# signal indicies: 0 = body acc x, 1 = body acc y, 2 = body acc z, 3 = body gyro x, 4 = body gyro y, 5 = body gyro z, 6 = total acc x, 7 = total acc y, 8 = total acc z
signal = 3  # signal index
sample = 1  # sample index of each label indicies list
scales = np.arange(1, 65)  # range of scales
wavelet = 'morl'  # mother wavelet

plot_cwt_coeffs_per_label(X_train, train_labels_indicies, LABEL_NAMES, signal, sample, scales, wavelet)

plt.show()