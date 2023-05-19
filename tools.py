import numpy as np
import scipy
import talib.abstract as ta
import talib
import math
import pandas as pd
import pandas_ta as pta

def midpoint(p1, p2):
	return p1 + (p2-p1) / 2

def distance(p1, p2):
	return abs(p1-p2)

def angle(what, lookback, mode, barsize = 1):
	"""
	What: dataframe kuri naudojam
	Lookback: kiek lyginam, static arba dynamic priklauso nuo mode
	Mode:
		1 - static lookback
		2 - dynamic lookback
	Barsize: dataframe barsize nustatymui, turim sukurt atr is high, low, close
	"""
	#ang = 57.2957795131 * np.arctan(abs(what - what.shift(lookback)))
	#ang = 180 * np.arctan(abs(what - what.shift(lookback)) / ta.ATR(high = dataframe['true_high'], low = dataframe['true_low'], close = dataframe['true_close'], timeperiod = 10) / lookback)
	#ang = 180 * np.arctan(abs(what - what.shift(lookback)) / abs(dataframe['true_close'].shift(1) - dataframe['true_open'].shift(1)) / lookback)

	df = pd.DataFrame(what).fillna(0)
	df['ang'] = 0
	df['delta_p'] = 0
	df['lk'] = lookback

	for i in range(len(what)):
		#df['tr'].iat[i] = max([df['true_high'].iat[i] - df['true_low'].iat[i], abs(df['true_high'].iat[i] - df['true_close'].iat[i-1]), abs(df['true_low'].iat[i] - df['true_close'].iat[i-1])])
		#df['tr'].iat[i] = max([df['high'].iat[i] - df['low'].iat[i], abs(df['high'].iat[i] - df['close'].iat[i-1]), abs(df['low'].iat[i] - df['close'].iat[i-1])])
		#df['ang'].iat[i] = 57.2957795131 * np.arctan((what.iat[i] - what.iat[i-lookback.iat[i]]) / df['tr'].iat[i])
		df['ang'].iat[i] = 57.2957795131 * np.arctan((what.iat[i] - what.iat[i-df['lk'].iat[i]]) / df['lk'].iat[i])
		if (mode == 2):
			if (df['lk'].iat[i] == 1):
				df['ang'].iat[i] = np.nan
		if(mode == 3):
			df['delta_p'].iat[i] = barsize.iat[i] * (what.iat[i] - what.iat[i-lookback])
			delta_t = 1
			df['ang'].iat[i] = math.atan2(df['delta_p'].iat[i], delta_t) * 180 / math.pi

	return df['ang']

def normalize(df, window, mode):
	"""
	Mode:
		1 = nuo 0..1
		2 = nuo -1..1
	"""
	df = df.fillna(0)
	df = (df - df.rolling(window).min()) / (df.rolling(window).max() - df.rolling(window).min())
	if (mode == 1):
		return df
	if (mode == 2):
		return (df - 0.5) * 2

def truechange(self, dataframe, what, mode):
	"""
	what: dataframe kuri naudojam

	Mode:
		1 = static procentinis
		2 = dinamiskas, pagal praeitus value
		3 = normalizuotas
	"""
	if (mode == 1):
		#b = ((a - a.shift(1)) / (a.shift(1))) * 100
		a = a + 1000
		b = (abs(a - a.shift(1)) / abs(a.shift(1))) * 100 * 1000
		return b

	if (mode == 2):
		df = dataframe.copy().fillna(0)
		df['a'] = what
		df['x'] = 0.0
		df['b'] = 0.0

		last_row = dataframe.tail(1).index.item()
		for i in range(self.startup_candle_count, last_row + 1):
			df['b'].iat[i] = ((df['a'].iat[i] - df['a'].iat[i-1]) / (df['a'].iat[i-1])) * 100
			df['x'].iat[i] = df['x'].iat[i-1] + df['b'].iat[i]
		return df['x']

	if (mode == 3):
		df = dataframe.copy().fillna(0)
		df['a'] = what
		df['b'] = 0.0
		df['x'] = 0.0
		df['z'] = 0.0
		df['c'] = formula(df, length = 288)

		last_row = dataframe.tail(1).index.item()
		for i in range(self.startup_candle_count, last_row + 1):
			df['b'].iat[i] = ((df['a'].iat[i] - df['a'].iat[i-1]) / (df['a'].iat[i-1])) * 100
			df['x'].iat[i] = np.where(
				(abs(df['x'].iat[i-1]) < df['c'].iat[i]), df['x'].iat[i-1] + df['b'].iat[i], -df['x'].iat[i])

			df['z'].iat[i] = np.where(
				((df['x'].iat[i] / df['c'].iat[i]) > 1) & ((df['x'].iat[i-1] / df['c'].iat[i-1]) < 1), -df['z'].iat[i], 
					np.where(
						((df['x'].iat[i] / df['c'].iat[i]) < -1) & ((df['x'].iat[i-1] / df['c'].iat[i-1]) > -1), -df['z'].iat[i], 
						np.where(
							(abs(df['z'].iat[i-1]) < df['c'].iat[i]), df['z'].iat[i-1] + df['b'].iat[i], -df['z'].iat[i])))
		return df['z'] / df['c']

	if (mode == 4):
		df = dataframe.copy().fillna(0)
		df['a'] = what
		df['c'] = formula(df, length = 144)
		df['x'] = 0.0
		df['b'] = 0.0

		last_row = dataframe.tail(1).index.item()
		for i in range(self.startup_candle_count, last_row + 1):
			df['b'].iat[i] = ((df['a'].iat[i] - df['a'].iat[i-1]) / (df['a'].iat[i-1])) * 100
			df['x'].iat[i] = df['x'].iat[i-1] + df['b'].iat[i]
		return df['x'] / df['c']

def truechange3(dataframe, what):
	df = dataframe.copy().fillna(0)
	C0 = 1.0
	#print(C0)
	C1 = 0.0
	tc = []
	for row in df.itertuples(index = True, name = 'close'):
		C0_1, C1_1 = C0, C1
		C0 = ((row.close - C0_1) / C0_1) * 100
		#print(row.close, C0)
		C1 = C1_1 + C0
		tc.append(C1)
	return tc

def meannormalize(df, window):
	df = df.fillna(0)
	df = (df - df.rolling(window).mean()) / (df.rolling(window).max() - df.rolling(window).min())
	return df

def tanhestimator(df, window):
	df = df.fillna(0)
	x = df.rolling(1).mean()
	m = df.rolling(window).mean()
	std = df.rolling(window).std()
	#xz = df.rolling(window).apply(scipy.stats.zscore)
	data = 0.5 * (np.tanh(0.01 * (x - m) / std) + 1)
	#data = 0.5 * (np.tanh(0.01 * (xz)) + 1)
	return data

def residuals(df, window):
	df = df - df.rolling(window).mean()
	return df

def velocity(dataframe, length = 20, k = 1):
	dataframe_max = dataframe.rolling(length).max()
	dataframe_min = dataframe.rolling(length).min()

	a = k * (dataframe - dataframe_min) / (dataframe_max - dataframe_min)
	return a

def expansion(a, b):
	c = np.where((a > a.shift(1)) & (b < b.shift(1)), 1, 0)
	return c

def gap(a, b):
	c = abs(a - b)
	return c

def ratio(a, b, mode):
	"""
	Mode:
		1: a/b lyginant su 1, return y,z
		2: a/b lyginant viena su kitu, return y
	"""
	if (mode == 1):
		x = abs(a) + abs(b)
		y = abs(a) / x
		z = abs(b) / x
		return y, z

	if (mode == 2):
		y = np.where(abs(a) > abs(b), abs(a) / abs(b), abs(b) / abs(a))
		return y

def counter(dataframe, a, b):
	"""
	a, b: trigger ir untrigger, paduoti .astype(int) (1,0)
	"""
	df = dataframe.copy().fillna(0)
	df['a'] = a
	df['b'] = b
	df['c'] = 1

	for i in range(len(df)):
		if ((df['a'].iat[i] == 1) & (df['b'].iat[i] == 0)):
			#print('triggered')
			df['c'].iat[i] = df['c'].iat[i-1] + 1

		if ((df['a'].iat[i] == 0) & (df['b'].iat[i] == 1)):
			#print('untriggered')
			df['c'].iat[i] = 1

		if ((df['c'].iat[i-1] > 1) & (df['b'].iat[i] == 0)):
			#print('counter+1')
			df['c'].iat[i] = df['c'].iat[i-1] + 1

	return df['c']

def trig(dataframe, a, b, mode):
	"""
	a,b:
		dataframe'ai kuriuos lyginam
	Mode: 
		0 = jei a, b floatai
		1 = jei a trinaris booleanas (1..0..-1), b == None
		2 = jei a ir b abu booleanai (1..0)
		3 = jei a (1..0) ir b (0..-1)
		4 = jei a > 0 ir a < 0
	"""
	df = dataframe.copy().fillna(0)
	df['x1'] = a
	df['x2'] = b

	df['trig'] = 0

	for i in range(0, len(df)):
		if (mode == 0):
			df['trig'].iat[i] = np.where(
				((df['trig'].iat[i-1] < 1) & (df['x1'].iat[i-1] < df['x2'].iat[i-1]) & (df['x1'].iat[i] > df['x2'].iat[i])), 1, 
				np.where(
					(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i-1] > df['x2'].iat[i-1]) & (df['x1'].iat[i] < df['x2'].iat[i]), -1, 
					np.where(
						((df['trig'].iat[i-1] == 1) & (df['x1'].iat[i] > df['x2'].iat[i])), 1, 
						np.where(
							((df['trig'].iat[i-1] == -1) & (df['x1'].iat[i] < df['x2'].iat[i])), -1, 0))))
		if (mode == 1):
			df['trig'].iat[i] = np.where(
				(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i-1] < 1) & (df['x1'].iat[i] > 0), 1, 
				np.where(
					(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i-1] > -1) & (df['x1'].iat[i] < 0), -1, 
					np.where(
						(df['trig'].iat[i-1] == 1) & (df['x1'].iat[i] > -1), 1, 
						np.where(
							(df['trig'].iat[i-1] == -1) & (df['x1'].iat[i] < 1), -1, 0))))
		if (mode == 2):
			df['trig'].iat[i] = np.where(
				(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == 1) & (df['x2'].iat[i] == 1), 1, 											#buvo -1 bet abu trig 1 vadinas 1
				np.where(
					(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == -1) & (df['x2'].iat[i] == -1), -1, 									#buvo 1 bet abu trig -1 vadinas -1
					np.where(
						(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == 1) & (df['x2'].iat[i] == -1), -1, 									#buvo -1 bet trig1 1 o trig2 -1 vadinas -1
						np.where(
							(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == -1) & (df['x2'].iat[i] == 1), -1, 								#buvo -1 bet trig1 -1 o trig2 1 vadinas -1
							np.where(
								(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == -1) & (df['x2'].iat[i] == 1), 1, 							#buvo 1 bet trig1 -1 o trig2 1 vadinas 1
								np.where(
									(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 1) & (df['x2'].iat[i] == -1), 1, 						#buvo 1 bet trig1 1 o trig2 -1 vadinas 1
									np.where(
										(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 1) & (df['x2'].iat[i] == 1), 1, 					#buvo 1 ir trig1 1 ir trig2 1 vadinas 1
										np.where(
											(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == -1) & (df['x2'].iat[i] == -1), -1, 0))))))))	#buvo -1 ir trig1 -1 ir trig2 -1 vadinas -1
		if (mode == 3):
			df['trig'].iat[i] = np.where(
				(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == 1) & (df['x2'].iat[i] == 0), 1, 											#buvo -1 bet trig1 1 o trig2 0 tai 1
				np.where(
					(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 0) & (df['x2'].iat[i] == -1), -1, 										#buvo 1 bet trig1 0 o trig2 -1 tai -1
						np.where(
							(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 0) & (df['x2'].iat[i] == 0), 1, 								#buvo 1 bet trig1 0 ir trig2 0 tai lieka 1
								np.where(
									(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == 0) & (df['x2'].iat[i] == 0), -1, 						#buvo -1 bet trig1 0 ir trig2 0 tai lieka -1
										np.where(
											(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 1) & (df['x2'].iat[i] == 0), 1, 				#buvo 1 bet trig1 1 ir trig2 0 tai 1
												np.where(
													(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == 0) & (df['x2'].iat[i] == -1), -1, 		#buvo -1 bet trig1 0 ir trig2 -1 tai -1
													0))))))
		if (mode == 4):
			df['trig'].iat[i] = np.where(
				(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] > 0), 1, 
				np.where(
					(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] < 0), -1, 
						np.where(
							(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 0), 1, 
								np.where(
									(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == 0), -1, 
										np.where(
											(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] > 0), 1, 
												np.where(
													(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] < 0), -1, 0))))))

	return df['trig']

def digitize(dataframe, a, sens, mode):
	"""
	a:
		dataframe kuri nauodam
	Sens:
		value pokycio jautrumas, pagal save
	Mode:
		1 = static
		2 = procentai
	"""
	df = dataframe.copy().fillna(0)
	df['x'] = a

	for i in range(len(df)):
		if (mode == 1):
			df['x'].iat[i] = np.where(
				(abs(df['x'].iat[i] - df['x'].iat[i-1])) > sens, df['x'].iat[i], df['x'].iat[i-1]
			)

		if (mode == 2):
			z = ((abs(df['x'].iat[i] - df['x'].iat[i-1])) * 100) / df['x'].iat[i-1]
			df['x'].iat[i] = np.where(
				(z > sens.iat[i]), df['x'].iat[i], df['x'].iat[i-1]
			)

	return df['x']

def formula(dataframe, length):
	#df = dataframe.copy().fillna(0)
	a = dataframe['close'].rolling(length).max()
	b = dataframe['open'].rolling(length).min()
	c = ((a - b) / b) * 100
	return c

def formula_min(dataframe, length):
	#df = dataframe.copy().fillna(0)
	a = (dataframe['close']*-1).rolling(length).max()
	b = (dataframe['open']*-1).rolling(length).min()
	c = ((a - b) / b) * 100
	return c

def formula2(dataframe, length):
	#df = dataframe.copy().fillna(0)
	a = dataframe['volume'].rolling(length).max()
	b = dataframe['volume'].rolling(length).min()
	c = ((a - b) / b) * 100
	return c

def sort_func(dataframe,listas):

	if (all(listas[i] >= listas[i+1] for i in range(len(listas) - 1))):
		return 1
	else:
		if (all(listas[i] <= listas[i+1] for i in range(len(listas) - 1))):
			return -1
		else:
			return 0

def conditioner(dataframe, what, condition):
	"""
	A sliding indicator that holds the value of a DataFrame's column
	until a given condition is met.
	"""
	df = dataframe.copy().fillna(0)
	df['value'] = 0
	
	for i in range(len(df)):
		if condition(df.iloc[i]):
			df.loc[df.index[i], 'value'] = what.iloc[i]
		if not condition(df.iloc[i]) and (df['value'].iloc[i] == 0):
			df.loc[df.index[i], 'value'] = df['value'].iloc[i - 1]
			
	return np.where(df['value'] > 0, df['value'], np.nan)

def sup_res(df, lookback):
	# Create empty columns for support and resistance levels
	df['sup'] = np.nan
	df['res'] = np.nan

	# Define the lookback periods
	period = lookback

	# Iterate over the dataframe
	for i in range(len(df)):
		# Get the current price
		current_price = df['close'].iloc[i]
		
		# Iterate over the lookback periods
		#for period in periods:
		# Calculate the minimum and maximum prices for the period
		min_price = df['low'].iloc[max(0, i-period+1):i+1].min()
		max_price = df['high'].iloc[max(0, i-period+1):i+1].max()
		
		# Calculate the support and resistance levels
		support = min_price	# - 0.001 * (max_price - min_price)
		resistance = max_price	# + 0.001 * (max_price - min_price)
		
		# Update the support and resistance columns if the current price is within the range
		if current_price >= support and current_price <= resistance:
			df['sup'].iloc[i] = support
			df['res'].iloc[i] = resistance

	return df['sup'],df['res']

def pivotas(dataframe, lookback = 200):
	"""
	Calculate pivot points using the highs and lows of a DataFrame.
	"""
	# Create empty DataFrames to store the pivot points
	pivot_highs = pd.DataFrame(index=dataframe.index, columns=['pivot_high'])
	pivot_lows = pd.DataFrame(index=dataframe.index, columns=['pivot_low'])
	
	for i in range(lookback, len(dataframe)):
		# Get the highs and lows for the lookback period
		high_slice = dataframe['high'].iloc[i-lookback:i]
		low_slice = dataframe['low'].iloc[i-lookback:i]
		
		# Find the highest and lowest values in the slice
		high = high_slice.max()
		low = low_slice.min()
		
		# If the current high is higher than the previous high, set the pivot high
		if high > high_slice.iloc[-2]:
			pivot_highs.loc[dataframe.index[i], 'pivot_high'] = high
		# Otherwise, copy the previous pivot high
		else:
			pivot_highs.loc[dataframe.index[i], 'pivot_high'] = pivot_highs.loc[dataframe.index[i-1], 'pivot_high']
		
		# If the current low is lower than the previous low, set the pivot low
		if low < low_slice.iloc[-2]:
			pivot_lows.loc[dataframe.index[i], 'pivot_low'] = low
		# Otherwise, copy the previous pivot low
		else:
			pivot_lows.loc[dataframe.index[i], 'pivot_low'] = pivot_lows.loc[dataframe.index[i-1], 'pivot_low']
	
	# Combine the pivot highs and lows into a single DataFrame and return it
	pivots = pd.concat([pivot_highs, pivot_lows], axis=1)
	return pivot_highs,pivot_lows

def extrema(dataframe):

	df = dataframe.copy()
	# Define window size for sliding indicator
	window_size = 5

	# Create a new column for local maxima and minima
	df['maxima'] = df['high'].rolling(window_size, center=True).max()
	df['minima'] = df['low'].rolling(window_size, center=True).min()

	# Iterate over the DataFrame and label the extremas
	for i in range(len(df)):
		if df.loc[i, 'high'] == df.loc[i+window_size, 'maxima']:
			df.loc[i, 'extrema'] = 'maxima'
		elif df.loc[i, 'low'] == df.loc[i+window_size, 'minima']:
			df.loc[i, 'extrema'] = 'minima'
		else:
			df.loc[i, 'extrema'] = 'none'

	# Display the DataFrame with extremas labeled
	return df['maxima'],df['minima']

def linear_growth(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
	time = max(0, trade_time - start_time)
	rate = (end - start) / (end_time - start_time)
	return min(end, start + (rate * time))


def detect_pullback(df, periods=30, method='pct_outlier'):
	"""     
	Pullback & Outlier Detection
	Know when a sudden move and possible reversal is coming
	
	Method 1: StDev Outlier (z-score)
	Method 2: Percent-Change Outlier (z-score)
	Method 3: Candle Open-Close %-Change
	
	outlier_threshold - Recommended: 2.0 - 3.0
	
	df['pullback_flag']: 1 (Outlier Up) / -1 (Outlier Down) 
	"""
	if method == 'stdev_outlier':
		outlier_threshold = 2.0
		df['dif'] = df['close'] - df['close'].shift(1)
		df['dif_squared_sum'] = (df['dif']**2).rolling(window=periods + 1).sum()
		df['std'] = np.sqrt((df['dif_squared_sum'] - df['dif'].shift(0)**2) / (periods - 1))
		df['z'] = df['dif'] / df['std']
		df['pullback_flag'] = np.where(df['z'] >= outlier_threshold, 1, 0)
		df['pullback_flag'] = np.where(df['z'] <= -outlier_threshold, -1, df['pullback_flag'])

	if method == 'pct_outlier':
		outlier_threshold = 2.0
		df['pb_pct_change'] = df['close'].pct_change()
		df['pb_zscore'] = qtpylib.zscore(df, window=periods, col='pb_pct_change')
		df['pullback_flag'] = np.where(df['pb_zscore'] >= outlier_threshold, 1, 0)
		df['pullback_flag'] = np.where(df['pb_zscore'] <= -outlier_threshold, -1, df['pullback_flag'])
	
	if method == 'candle_body':
		pullback_pct = 1.0
		df['change'] = df['close'] - df['open']
		df['pullback'] = (df['change'] / df['open']) * 100
		df['pullback_flag'] = np.where(df['pullback'] >= pullback_pct, 1, 0)
		df['pullback_flag'] = np.where(df['pullback'] <= -pullback_pct, -1, df['pullback_flag'])
	
	return df