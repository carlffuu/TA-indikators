import pandas as pd
import numpy as np
import talib.abstract as ta
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors

def lorentzian_distance(x, y, gamma=1):
	return np.log(1 + gamma * np.abs(x - y))

def lorentzian_distances(dataframe, what):
	df = dataframe.copy().fillna(0)
	print('what length = ',len(what))
	for i in range(len(df)):
		x = 0
		for x in range(len(what)):
			print(i,'-',what[x].iloc[i])

def get_sum_of_distances(dataframe, a, b, c, gamma=1):
	df = dataframe.copy().fillna(0)
	#Compute the Lorentzian distances
	df['ab'] = lorentzian_distance(a, b, gamma)
	df['ac'] = lorentzian_distance(a, c, gamma)
	df['bc'] = lorentzian_distance(b, c, gamma)
	df['sum_of_distances'] = 0
	#Compute the sum of the distances
	for i in range(len(df)):
		df['sum_of_distances'].iat[i] = np.sum(df['ab'].iat[i]) + np.sum(df['ac'].iat[i]) + np.sum(df['bc'].iat[i])
	return df['sum_of_distances']

def get_sum_of_distances2(dataframe, a, b, c, gamma=1):
	df = dataframe.copy().fillna(0)
	#Compute the Lorentzian distances
	df['aa'] = 0
	df['bb'] = 0
	df['cc'] = 0
	df['sum_of_distances'] = 0
	#Compute the sum of the distances
	for i in range(len(df)):
		df['aa'].iat[i] = lorentzian_distance(a.iat[i], a.iat[i-1], gamma)
		df['bb'].iat[i] = lorentzian_distance(b.iat[i], b.iat[i-1], gamma)
		df['cc'].iat[i] = lorentzian_distance(c.iat[i], c.iat[i-1], gamma)
		df['sum_of_distances'].iat[i] = np.sum(df['aa'].iat[i]) + np.sum(df['bb'].iat[i]) + np.sum(df['cc'].iat[i])
	return df['sum_of_distances']

#WaveTrend 3D

def dual_pole_filter(dataframe, source, lookback = 9):
	df = dataframe.copy().fillna(0)
	source = source.fillna(0)
	omega = -99 * np.pi / (70 * lookback)
	alpha = np.exp(omega)
	beta = -(alpha**2)
	gamma = np.cos(omega) * 2 * alpha
	delta = 1 - gamma - beta
	#sliding_avg = 0.5 * (source + source.shift(1))

	df['sliding_avg'] = 0
	df['filter'] = 0
	for i in range (len(df)):
		df['sliding_avg'] = 0.5 * (source.iat[i] + source.iat[i-1])
		df['filter'].iat[i] = (delta * df['sliding_avg'].iat[i]) + (gamma * df['filter'].iat[i-1]) + (beta * df['filter'].iat[i-2])
	return df['filter']

def tanh(a):
	return -1 + 2 / (1 + np.exp(-2*a))

def normalize_deriv(dataframe, source, window):
	df = dataframe.copy().fillna(0)
	df['derivative'] = 0
	df['derivative_sum'] = 0
	df['quadratic_mean'] = 0

	for i in range(len(df)):
		df['derivative'].iat[i] = source.iat[i] - source.iat[i-2]
		df['derivative_sum'].iat[i] = ((df['derivative']**2).rolling(window).sum()).iat[i]
		#for j in range(len(df) - window):
			#df['derivative_sum'].iat[j] += df['derivative_sum'].iat[j] + df['derivative'].iat[j - 1] 
		df['quadratic_mean'].iat[i] = np.sqrt((df['derivative_sum'].iat[i]) / window)
	return df['derivative']/df['quadratic_mean']

def oscillator(dataframe, source, smooth, window):
	nderiv = normalize_deriv(dataframe, source, window)
	hyperbolic_tanh = tanh(nderiv)
	return dual_pole_filter(dataframe, hyperbolic_tanh, lookback = smooth)

#######

def rescale(what, old_min, old_max, new_min, new_max):
	return new_min + (new_max - new_min) * (what - old_min) / (old_max - old_min)

def volatility_filter(dataframe, fast, slow):
	fast_atr = ta.ATR(dataframe, fast)
	slow_atr = ta.ATR(dataframe, slow)
	return np.where(fast_atr > slow_atr, 1, -1)

def sliding_cumulative_volatility(src: pd.Series, window: int, scale_min: float = -1.0, scale_max: float = 10.0) -> pd.Series:
	atr = ta.ATR(src, window)
	cum_atr = atr.rolling(window).sum()
	norm_cum_atr = (cum_atr - cum_atr.min()) / (cum_atr.max() - cum_atr.min()) * (scale_max - scale_min) + scale_min
	return norm_cum_atr

def sliding_atr_oscillator(data: pd.DataFrame, window: int) -> pd.Series:
	atr = ta.ATR(data['high'], data['low'], data['close'], timeperiod=window)
	di_minus = ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=window)
	cv = atr * di_minus / 100
	cv_cum = pd.Series(cv).rolling(window).sum()
	max_cv_cum = cv_cum.rolling(window=window, min_periods=1).max()
	min_cv_cum = cv_cum.rolling(window=window, min_periods=1).min()
	cv_osc = (2 * (cv_cum - min_cv_cum) / (max_cv_cum - min_cv_cum) - 1)
	return cv_osc

def ehlers_regime_filter(dataframe) -> pd.Series:
	df = dataframe.copy().fillna(0)
	df['ohlc'] = (df['open']+df['high']+df['low']+df['close']) / 4

	value1 = pd.Series(0.0, index=df.index)
	value2 = pd.Series(0.0, index=df.index)
	klmf = pd.Series(0.0, index=df.index)
	absCurveSlope = pd.Series(0.0, index=df.index)
	exponentialAverageAbsCurveSlope = pd.Series(0.0, index=df.index)
	normalized_slope_decline = pd.Series(0.0, index=df.index)

	for i in range(1, len(df)):
		value1.iloc[i] = 0.2 * (df['ohlc'].iloc[i] - df['ohlc'].iloc[i-1]) + 0.8 * np.nan_to_num(value1.iloc[i-1])
		value2.iloc[i] = 0.1 * (df['high'].iloc[i] - df['low'].iloc[i]) + 0.8 * np.nan_to_num(value2.iloc[i-1])
		omega = np.abs(value1.iloc[i] / value2.iloc[i])
		alpha = (-np.power(omega,2) + np.sqrt(np.power(omega, 4) + 16 * np.power(omega,2))) / 8 
		klmf.iloc[i] = alpha * df['ohlc'].iloc[i] + (1 - alpha) * np.nan_to_num(klmf.iloc[i-1])
		absCurveSlope.iloc[i] = np.abs(klmf.iloc[i] - klmf.iloc[i-1])
		exponentialAverageAbsCurveSlope.iloc[i] = pd.Series.ewm(absCurveSlope.iloc[:i+1], span=200).mean().iloc[-1]
		normalized_slope_decline.iloc[i] = (absCurveSlope.iloc[i] - exponentialAverageAbsCurveSlope.iloc[i]) / exponentialAverageAbsCurveSlope.iloc[i]
	return normalized_slope_decline

def weighted_sliding_average(source, window_size=5, sigma=1):
	#Compute the weights for the radial basis function
	weights = np.zeros(window_size)
	for i in range(window_size):
		weights[i] = np.exp(-(i ** 2) / (2 * sigma ** 2))

	#Normalize the weights so that they sum to 1
	weights /= np.sum(weights)

	#Compute the weighted sliding average
	result = np.zeros_like(source)
	for i in range(window_size, len(source)):
		#Compute the weighted sum of the values in the window
		window_sum = 0
		for j in range(window_size):
			window_sum += weights[j] * source[i - window_size + j]
		#Store the weighted average in the result array
		result[i] = window_sum
	return result

def knn(dataframe, window):#, a, b, c):
	df = dataframe.fillna(0)
	data = pd.read_csv('./user_data/training_data.csv', delimiter = ';')
	model = KNeighborsRegressor(n_neighbors=8, weights='distance', metric='euclidean')
	# X = df[[a,b,c]]
	# X.columns = [[a,b,c]]
	# Y = df['close']
	# X_train = X[:-window]
	# Y_train = Y[:-window]
	A = df[['w3d_fast_signal_rescaled','w3d_norm_signal_rescaled','w3d_slow_signal_rescaled','norm_rsi_14','norm_cci_14']]
	X = data[['w3d_fast_signal_rescaled','w3d_norm_signal_rescaled','w3d_slow_signal_rescaled','norm_rsi_14','norm_cci_14']]
	X.columns = [['w3d_fast_signal_rescaled','w3d_norm_signal_rescaled','w3d_slow_signal_rescaled','norm_rsi_14','norm_cci_14']]
	Y = data['norm_close']

	X_train = X[:-window]
	Y_train = Y[:-window]

	model.fit(X_train, Y_train)
	predictions = []
	for i in range(window,len(df)):
		X_test = A.iloc[i]
		prediction = model.predict(X_test.values.reshape(1, -1))
		predictions.append(prediction[0])
	index = df.index[window:]
	return pd.Series(predictions, index=index)
