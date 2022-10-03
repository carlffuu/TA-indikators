import numpy as np
import talib.abstract as ta
import math
import pandas as pd
import pandas_ta as pta
from pandas import DataFrame, Series

from scipy import integrate

#Tools

def midpoint(p1, p2):
	return p1 + (p2-p1) / 2

def distance(p1, p2):
	return abs(p1-p2)

def angle(what, lookback):
	#rad2degree = 180 / 3.141592653589793238462643
	rad2degree = 57.2957795131
	ang = rad2degree * np.arctan(what - what.shift(lookback))
	return ang / 10

def normalize(df, window, mode):
	"""
	Mode:
		1 = nuo 0..1
		2 = nuo -1..1
	"""
	df = (df - df.rolling(window).min()) / (df.rolling(window).max() - df.rolling(window).min())
	if (mode == 1):
		return df
	if (mode == 2):
		return (df - 0.5) * 2

def meannormalize(df, window):
	df = (df - df.rolling(window).mean()) / (df.rolling(window).max() - df.rolling(window).min())
	return df

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

def ratio(a, b):
	x = abs(a) + abs(b)
	y = abs(a) / x
	z = abs(b) / x
	return y, z

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
				(df['trig'].iat[i-1] < 1) & (df['x1'].iat[i] == 1) & (df['x2'].iat[i] == 0), 1,												#buvo -1 bet trig1 1 o trig2 0 tai 1
				np.where(
					(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 0) & (df['x2'].iat[i] == -1), -1, 										#buvo 1 bet trig1 0 o trig2 -1 tai -1
						np.where(
							(df['trig'].iat[i-1] > -1) & (df['x1'].iat[i] == 0) & (df['x2'].iat[i] == 0), 1,								#buvo 1 bet trig1 0 ir trig2 0 tai lieka 1
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

def digitize(dataframe, a, sens):
	"""
	a:
		dataframe kuri nauodam
	Sens:
		value pokycio jautrumas, pagal save
	"""
	df = dataframe.copy().fillna(0)
	df['x'] = a
	df['sum'] = 0
	for i in range(len(df)):
		df['x'].iat[i] = np.where(
			(abs(df['x'].iat[i] - df['x'].iat[i-1])) > sens, df['x'].iat[i], df['x'].iat[i-1]
		)
	return df['x']

def truechange(a):
	b = ((a - a.shift(1)) / (a.shift(1))) * 1000
	return b

#Indicators

def polynomial_regression(df, what, ssf_length, rolling_window, mult):
	"""
	What:
		dataframe kuri naudojam
	SSF_length:
		super smoother filtro ilgis
	Rolling_window:
		zvakiu kiekis kuriam pritaikysim std
	Mult:
		std bands'u plotis
	"""
	dfcopy = df.copy()
	df = pd.DataFrame(index = dfcopy.index)

	def gradientas(y: pd.Series, window: int = rolling_window) -> pd.Series:
		x = np.arange(rolling_window)
		gradient, _ = np.polynomial.Polynomial.fit(x, y, deg=1)
		return gradient

	df['ssf_filter'] = pta.ssf(what, length = ssf_length, poles = 3)
	df['ssf_poly'] = df['ssf_filter'].rolling(rolling_window).apply(gradientas, raw = False)
	df['stdev'] = df['ssf_poly'].rolling(rolling_window, min_periods = 1).apply(np.std) * mult
	df['ssf_poly_u'] = df['ssf_poly'] + df['stdev']
	df['ssf_poly_l'] = df['ssf_poly'] - df['stdev']

	return df
	#return dataframe['ssf'],dataframe['ssf_poly'],dataframe['ssf_poly_u'],dataframe['ssf_poly_l']

def hoffman_ma(dataframe):
	"""
	Strategy - b auksciau uz c, tarp b ir c neturi but jokiu kitu MA
	"""
	dataframe['b'] = ta.SMA(dataframe,timeperiod = 5) 		#slow speed 
	
	dataframe['c'] = ta.EMA(dataframe,timeperiod = 18) 		#slow speed line
	dataframe['d'] = ta.EMA(dataframe,timeperiod = 20) 		#slow primary trend line

	dataframe['e'] = ta.SMA(dataframe,timeperiod = 50) 		#trend line 1
	dataframe['f'] = ta.SMA(dataframe,timeperiod = 89) 		#trend line 2
	dataframe['g'] = ta.EMA(dataframe,timeperiod = 144) 	#trend line 3

	dataframe['k'] = ta.EMA(dataframe, timeperiod = 35) 	#no trend zone - midline
	dataframe['tr'] = ta.TRANGE(dataframe)
	dataframe['r'] = pta.rma(dataframe['tr'],35) 			#no trend zone - midline
	dataframe['ku'] = dataframe['k'] + dataframe['r']*0.5 	#no trend zone - upperline
	dataframe['kl'] = dataframe['k'] - dataframe['r']*0.5 	#no trend zone - lowerline

	return dataframe['b'],dataframe['c'],dataframe['d'],dataframe['e'],dataframe['f'],dataframe['g'],dataframe['k'],dataframe['tr'],dataframe['r'],dataframe['ku'],dataframe['kl']

def hoffman_inv_retracement(dataframe, percentage, mode):
	"""
	Percentage:
		open, close , high, low lyginimas procentaliai
	Mode:
		1 = grazina  1, -1, 0
		2 = grazina tik 1, 0
	"""
	a = abs(dataframe['high'] - dataframe['low'])
	b = abs(dataframe['close'] - dataframe['open'])

	percentage = percentage / 100
	rv = (b < percentage * a).astype(int)

	x = dataframe['low'] + (a * percentage)
	y = dataframe['high'] - (a * percentage)

	sl = np.where((rv == 1) & (dataframe['high'] > y) & (dataframe['close'] < y) & (dataframe['open'] < y), 1, 0)
	ss = np.where((rv == 1) & (dataframe['low'] < x) & (dataframe['close'] > x) & (dataframe['open'] > x), 1, 0)
	if (mode == 1):
		hoff = np.where(
			(sl == 1) & (ss == 0), 1, 
				np.where(
					(ss == 1) & (sl == 0), -1, 0))
	if (mode == 2):
		hoff = np.where(
			((sl == 1) & (ss == 0)) | ((ss == 1) & (sl == 0)), 1, 0)

	return hoff

def FVE(dataframe, length, factor):
	"""
	https://www.tradingview.com/script/DWEa0qFz-Finite-Volume-Elements-FVE-Strategy/
	"""
	df = dataframe.copy().fillna(0)
	df['SMAV'] = ta.SMA(df['volume'], timeperiod = length)
	df['NMF'] = df['close'] - df['hl2'] + df['hlc3'] - df['hlc3'].shift(1)
	df['NVLM'] = np.where(
		(df['NMF'] > factor * df['close'] / 100), df['volume'], 
		np.where(
		(df['NMF'] < -factor * df['close'] / 100), -df['volume'], 0)
		)

	df['NRES'] = 0.0

	for i in range(length, len(df)):
		df['NRES'].iat[i] = df['NRES'].iat[i-1] + ((df['NVLM'].iat[i] / df['SMAV'].iat[i]) / length) * 100

	return df['NRES']

def coral_trend_indicator(dataframe, what, length, d):
	"""
	https://www.tradingview.com/script/AzQo1gRi-Coral-Trend-Indicator-LazyBear-pine-v4/
	"""
	df = dataframe.copy().fillna(0)
	df['x1'] = what

	di = (length - 1.0) / 2.0 + 1.0
	c1 = 2 / (di + 1.0)
	c2 = 1 - c1
	c3 = 3.0 * (d * d + d * d * d)
	c4 = -3.0 * (2.0 * d * d + d + d * d * d)
	c5 = 3.0 * d + 1.0 + d * d * d + 3.0 * d * d

	df['bfr'] = df['i1'] = df['i2'] = df['i3'] = df['i4'] = df['i5'] = df['i6'] = 0.0

	for i in range(len(df)):
		df['i1'].iat[i] = c1*df['x1'].iat[i] + c2*(df['i1'].iat[i-1])
		df['i2'].iat[i] = c1*df['i1'].iat[i] + c2*(df['i2'].iat[i-1])
		df['i3'].iat[i] = c1*df['i2'].iat[i] + c2*(df['i3'].iat[i-1])
		df['i4'].iat[i] = c1*df['i3'].iat[i] + c2*(df['i4'].iat[i-1])
		df['i5'].iat[i] = c1*df['i4'].iat[i] + c2*(df['i5'].iat[i-1])
		df['i6'].iat[i] = c1*df['i5'].iat[i] + c2*(df['i6'].iat[i-1])
		df['bfr'].iat[i] = -d*d*d*df['i6'].iat[i] + c3*(df['i5'].iat[i]) + c4*(df['i4'].iat[i]) + c5*(df['i3'].iat[i])

	return df['bfr']

def laguerre(dataframe, what, gamma):
	"""
	https://www.tradingview.com/script/JwaPW9af-Laguerre-Multi-Filter-DW/
	"""
	df = dataframe
	g = gamma

	laguerred = []
	L0, L1, L2, L3 = 0.0, 0.0, 0.0, 0.0
	for row in df.itertuples(index = True, name = 'close'):
		L0_1, L1_1, L2_1, L3_1 = L0, L1, L2, L3
		L0 = (1 - g) * getattr(row, what) + g * L0_1
		L1 = -g * L0 + L0_1 + g * L1_1
		L2 = -g * L1 + L1_1 + g * L2_1
		L3 = -g * L2 + L2_1 + g * L3_1
		LagF = (L0 + 2 * L1 + 2 * L2 + L3) / 6
		laguerred.append(LagF)

	return laguerred

def hull(dataframe, timeperiod):
	if isinstance(dataframe, Series):
		return ta.WMA(2 * ta.WMA(dataframe, int(math.floor(timeperiod / 2))) - ta.WMA(dataframe, timeperiod), int(round(np.sqrt(timeperiod))))
	else:
		return ta.WMA(2 * ta.WMA(dataframe[f'close'], int(math.floor(timeperiod / 2))) - ta.WMA(dataframe[f'close'], timeperiod), int(round(np.sqrt(timeperiod))))

def zlsma(dataframe, length):
	df = dataframe.copy()
	df['lsma'] = ta.LINEARREG(df, timeperiod = length)
	df['lsma2'] = ta.LINEARREG(df['lsma'], timeperiod = length)
	eq = df['lsma'] - df['lsma2']
	df['zlsma'] = df['lsma2'] + eq
	zlsmaboi = pd.DataFrame(df['zlsma']).fillna(0)
	return zlsmaboi

def VWAP(high, low, close, volume, length = 200):
	typical = ((high + low + close) / 3)
	left = (volume * typical).rolling(window = length, min_periods = length).sum()
	right = volume.rolling(window = length, min_periods = length).sum()
	return left / right
	#return pd.Series(index = bars.index, data = (left / right)).replace([np.inf, -np.inf], float('NaN')).ffill()

def CVD(dataframe):
	dataframe['upper_wick'] = np.where(dataframe['close'] > dataframe['open'], dataframe['high'] - dataframe['close'], dataframe['high'] - dataframe['open'])
	dataframe['lower_wick'] = np.where(dataframe['close'] > dataframe['open'], dataframe['open'] - dataframe['low'], dataframe['close'] - dataframe['low'])
	dataframe['spread'] = dataframe['high'] - dataframe['low']
	dataframe['body_length'] = dataframe['spread'] - (dataframe['upper_wick'] + dataframe['lower_wick'])
	dataframe['percent_upper_wick'] = dataframe['upper_wick'] / dataframe['spread']
	dataframe['percent_lower_wick'] = dataframe['lower_wick'] / dataframe['spread']
	dataframe['percent_body_length'] = dataframe['body_length'] / dataframe['spread']
	dataframe['buying_volume'] = np.where(dataframe['close'] > dataframe['open'], ((dataframe['percent_body_length'] + (dataframe['percent_upper_wick'] + dataframe['percent_lower_wick']) / 2) * dataframe['volume']), (((dataframe['percent_upper_wick'] + dataframe['percent_lower_wick']) / 2) * dataframe['volume']))
	dataframe['selling_volume'] = np.where(dataframe['close'] < dataframe['open'], ((dataframe['percent_body_length'] + (dataframe['percent_upper_wick'] + dataframe['percent_lower_wick']) / 2) * dataframe['volume']), (((dataframe['percent_upper_wick'] + dataframe['percent_lower_wick']) / 2) * dataframe['volume']))
	return dataframe['buying_volume'], dataframe['selling_volume']

#Monotonic calculations

def monotonic_simple(what, sens, mode):
	"""
	What:
		dataframe kuri naudojam
	Mode:
		1 = returnina up ir down
		2 = returnina viena reiksme
	Sens:
		jautrumas, nustatom pagal save
	"""
	a = np.where((what > what.shift(1)) & (abs(what-what.shift(1)) > sens), 1, 0) 	#mono dir up
	b = np.where((what < what.shift(1)) & (abs(what-what.shift(1)) > sens), -1, 0) 	#mono dir dn
	if (mode == 1):
		return a, b
	if (mode == 2):
		c = np.where(a == 1, 1, np.where(b == -1, -1 ,0))
		return c

def monotonic_change(what, sens, mode):
	"""
	What:
		dataframe kuri naudojam
	Mode:
		1 = returnina up ir down
		2 = returnina viena reiksme
	Sens:
		jautrumas, nustatom pagal save
	"""
	a = np.where((what > what.shift(1)) & (truechange(what) > sens), 1, 0) 	#mono dir up
	b = np.where((what < what.shift(1)) & (truechange(what) < sens), -1, 0) #mono dir dn
	if (mode == 1):
		return a, b
	if (mode == 2):
		c = np.where(a == 1, 1, np.where(b == -1, -1 ,0))
		return c

def monotonic_advanced(what, length, thresh_p, thresh_n, window, sens):
	"""
	What:
		dataframe kuri naudojam
	Length:
		kiek zingsniu atgal sumuosim a
	Thresh_p, Thresh_n
		kartele dataframe'ui a
	Window:
		koki kieki informacijos normalizuojam
	Sens:
		jautrumas, nustatom pagal save
	"""
	a = what.rolling(length).sum()
	a = normalize(a, window = window, mode = 1) 	#mono line normalizuotas tarp 0..1

	b = np.where((a > (thresh_p / 100)), 1, 0) 		#mono thresh long
	c = np.where((a < (thresh_n / 100)), -1, 0)		#mono thresh short

	d = np.where(((a > a.shift(1)) & (abs(a-a.shift(1)) > sens)) | (a == 1), 1, 0) 	#mono dir up
	e = np.where(((a < a.shift(1)) & (abs(a-a.shift(1)) > sens)) | (a == 0), -1, 0) #mono dir dn

	if (thresh_p > 0) & (thresh_n > 0):
		return a, b, c, d, e
	if (thresh_p > 0) & (thresh_n == 0):
		return a, b, d, e
	if (thresh_p == 0) & (thresh_n > 0):
		return a, c, d, e

#MA resonance

def rez_ma(close, high, low, volume, df, vel_length = 20, ma_list = [], ma_type = 'ssf', thresh = 0, mono_sens = 0, pref = ''):
	data = []
	dfcopy = df.copy()
	df = pd.DataFrame(index = dfcopy.index)

	def ma_velo(df, vel_length = vel_length, k = 1):
		df_max = df.rolling(vel_length).max()
		df_min = df.rolling(vel_length).min()
		a = k * (df - df_min) / (df_max - df_min)
		return a
	ma_e = 0
	ma_num = range(0, len(ma_list))
	for i, j in zip(ma_list, ma_num):
		if ma_type == 'ssf':
			df[f'{ma_type}_{pref}-{j}'] = pta.ssf(close, length = i, poles = 3)
		if ma_type == 't3':
			df[f'{ma_type}_{pref}-{j}'] = pta.t3(close, length = i)
		if ma_type == 'hull':
			df[f'{ma_type}_{pref}-{j}'] = pta.hma(close, length = i)
		if ma_type == 'vwma':
			df[f'{ma_type}_{pref}-{j}'] = pta.vwma(close, volume, length = i)
		if ma_type == 'swma':
			df[f'{ma_type}_{pref}-{j}'] = pta.swma(close, length = i)
		if ma_type == 'kama':
			df[f'{ma_type}_{pref}-{j}'] = pta.kama(close, length = i)
		if ma_type == 'vhf':
			df[f'{ma_type}_{pref}-{j}'] = pta.vhf(close, length = i)
		if ma_type == 'zlsma':
			df[f'{ma_type}_{pref}-{j}'] = zlsma(close, length = i)
		if ma_type == 'vwap':
			df[f'{ma_type}_{pref}-{j}'] = VWAP(high, low, close, volume, length = i)
		ma_e += df[f'{ma_type}_{pref}-{j}']

	#### H
	df[f'{ma_type}_{pref}'] = ma_e / len(ma_list)
	df[f'{ma_type}_{pref}-trend'] = np.where((df[f'{ma_type}_{pref}'] > 0), 1, -1)

	#### Velocity
	df[f'{ma_type}_{pref}-vel'] = ma_velo(df[f'{ma_type}_{pref}'], vel_length = vel_length, k = 1)
	df[f'{ma_type}_{pref}-vel_trend'] = np.where(
		(df[f'{ma_type}_{pref}-vel'] == 1), 1, 
		np.where(
		(df[f'{ma_type}_{pref}-vel'] == 0), -1, 0))

	df[f'{ma_type}_{pref}-vel_dir'] = np.where(df[f'{ma_type}_{pref}-vel'] > df[f'{ma_type}_{pref}-vel'].shift(1), 1, -1)
	df[f'{ma_type}_{pref}-vel_dir+'] = np.where(
		(df[f'{ma_type}_{pref}-vel'] > 0.95) | (df[f'{ma_type}_{pref}-vel_dir'] == 1), 1, 
		(np.where((df[f'{ma_type}_{pref}-vel'] < 0.05) | (df[f'{ma_type}_{pref}-vel_dir'] == -1), -1, 0)))

	#### Angle
	df[f'{ma_type}_{pref}-angle'] = ta.LINEARREG_ANGLE(df[f'{ma_type}_{pref}'])
	df[f'{ma_type}_{pref}-angle_trend'] = np.where((df[f'{ma_type}_{pref}-angle'] > 0), 1, -1)

	#### Mono
	df[f'{ma_type}_{pref}-mono_line'], df[f'{ma_type}_{pref}-mono_thr_long'], df[f'{ma_type}_{pref}-mono_thr_short'], df[f'{ma_type}_{pref}-mono_dir_up'], df[f'{ma_type}_{pref}-mono_dir_dn'] = monotonic_advanced(what = df[f'{ma_type}_{pref}'], length = 6, thresh_p = thresh, thresh_n = 100-thresh, window = 700, sens = mono_sens)

	df[f'{ma_type}_{pref}-mono_2_dir_up'], df[f'{ma_type}_{pref}-mono_2_dir_dn'] = monotonic_simple(what = df[f'{ma_type}_{pref}'], sens = mono_sens, mode = 1)

	df[f'{ma_type}_{pref}-angle_mono_dir_up'], df[f'{ma_type}_{pref}-angle_mono_dir_dn'] = monotonic_simple(what = df[f'{ma_type}_{pref}-angle'], sens = 0.12, mode = 1)

	#### Correl
	df[f'{ma_type}_{pref}-correl_close'] = ta.CORREL(df[f'{ma_type}_{pref}'], close, timeperiod = 20)
	mean_volume = (meannormalize(volume, window = 150).fillna(0)*2)
	df[f'{ma_type}_{pref}-correl_vol'] = ta.CORREL(df[f'{ma_type}_{pref}'], mean_volume, timeperiod = 20)
	
	#### TSF
	df[f'{ma_type}_{pref}-tsf'] = ta.BETA(high, low, timeperiod = 20)

	print_that = 0
	if print_that == 1:
		tab = "\N{TAB}"
		print(f'{tab}#{str(ma_type)}')
		for col in df.columns:
			if col.startswith(str(ma_type)):
				print(f"{tab}'{col}'")

	return df