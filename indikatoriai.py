import numpy as np
import talib.abstract as ta
import math
import pandas as pd
import pandas_ta as pta
from pandas import DataFrame, Series

def engulf2(dataframe):
	df = dataframe.copy().fillna(0)
	df['bull'] = 0
	df['bear'] = 0
	df['rsi'] = 0
	df['ob'] = 0
	df['os'] = 0

	df['rsi'] = ta.RSI(df['close'],14)

	for i in range(len(df)):
		df['bull'].iat[i] = np.where((df['close'].iat[i] >= df['open'].iat[i-1]) and (df['close'].iat[i-1] < df['open'].iat[i-1]), 1, 0)
		df['bear'].iat[i] = np.where((df['close'].iat[i] <= df['open'].iat[i-1]) and (df['close'].iat[i-1] > df['open'].iat[i-1]), 1, 0)

		
		df['ob'].iat[i] = np.where(df['rsi'].iat[i] >= 70, 1, 0)
		df['os'].iat[i] = np.where(df['rsi'].iat[i] <= 30, 1, 0)

	return df['bull'], df['bear'], df['ob'], df['os']

def engulf(a,b):

	dataframe['rsi'] = ta.RSI(a,14)

	dataframe['bull'] = np.where((a >= b.shift(1)) and (a.shift(1) < b.shift(1)), 1, 0)
	dataframe['bear'] = np.where((a <= b.shift(1)) and (a.shift(1) > b.shift(1)), 1, 0)

	dataframe['ob'] = np.where(dataframe['rsi'] >= 70, 1, 0)
	dataframe['os'] = np.where(dataframe['rsi'] <= 30, 1, 0)

	return dataframe['bull'], dataframe['bear'], dataframe['ob'], dataframe['os']

def funcLinearRegressionChannel2(dtloc, source = 'close', window = 180, deviations = 2): 
	dtLRC = dtloc.copy()
	dtLRC['lrc_up'] = np.nan
	dtLRC['lrc_down'] = np.nan
	i = np.arange(window)
	i = i[::-1]
	Ex = i.sum()
	Ex2 = (i * i).sum()
	ExT2 = math.pow(Ex, 2)
	def calc_lrc(dfr, init=0):
		global calc_lrc_src_value
		if init == 1:
			calc_lrc_src_value = list()
			return
		calc_lrc_src_value.append(dfr[source])
		lrc_val_up = np.nan
		lrc_val_down = np.nan
		intercept = 0
		slope = 0
		if len(calc_lrc_src_value) > window:
			calc_lrc_src_value.pop(0)
		if len(calc_lrc_src_value) >= window:
			src = np.array(calc_lrc_src_value)
			Ey = src.sum()
			Ey2 = (src * src).sum()
			EyT2 = math.pow(Ey,2)
			Exy = (i*src).sum()
			PearsonsR = (Exy - Ex * Ey / window) / (math.sqrt(Ex2 - ExT2 / window) * math.sqrt(Ey2 - EyT2 / window))
			ExEx = Ex * Ex
			slope = 0.0
			if (Ex2 != ExEx ):
				slope = (window * Exy - Ex * Ey) / (window * Ex2 - ExEx)
			linearRegression = (Ey - slope * Ex) / window
			intercept = linearRegression + window * slope
			deviation = np.power((src - (intercept - slope * (window - i))), 2).sum()
			devPer = deviation / window
			devPerSqrt = math.sqrt(devPer)
			deviation = deviations * devPerSqrt
			lrc_val_up = linearRegression + deviation
			lrc_val_down = linearRegression - deviation
		return lrc_val_up,lrc_val_down, intercept, slope
	calc_lrc(None, init=1)
	dtLRC[['lrc_up','lrc_down','intercept','slope']] = dtLRC.apply(calc_lrc, axis = 1, result_type='expand')
	#return dtLRC[['lrc_up','lrc_down']]
	return dtLRC['lrc_up'],dtLRC['lrc_down'],dtLRC['intercept'],dtLRC['slope']

def LinearRegressionChannel2(dtloc, source = 'close', window = 180, deviations = 2):
	dtLRC = dtloc.copy()
	dtLRC['lrc_up'] = np.nan
	dtLRC['lrc_down'] = np.nan
	colSource = dtLRC.loc[:, source].values
	collrc_up = dtLRC.loc[:, 'lrc_up'].values
	collrc_down = dtLRC.loc[:, 'lrc_down'].values
	Ex = 0.0
	Ey = 0.0
	Ex2 = 0.0
	Ey2 = 0.0
	Exy = 0.0
	for i in range(window):
		closeI = colSource[-(i+1)]
		Ex = Ex + i 
		Ey = Ey + closeI
		Ex2 = Ex2 + i*i
		Ey2 = Ey2 + closeI*closeI
		Exy = Exy + i*closeI
	ExT2 = math.pow(Ex,2)
	EyT2 = math.pow(Ey,2)
	PearsonsR = (Exy - Ex * Ey / window) / (math.sqrt(Ex2 - ExT2 / window) * math.sqrt(Ey2 - EyT2 / window))
	ExEx = Ex * Ex
	slope = 0.0
	if (Ex2 != ExEx ):
		slope = (window * Exy - Ex * Ey) / (window * Ex2 - ExEx)
	linearRegression = (Ey - slope * Ex) / window
	intercept = linearRegression + window * slope
	deviation = 0.0
	for i in range(window):
		deviation = deviation + math.pow((colSource[-(i+1)] - (intercept - slope * (window - i))), 2)
	devPer = deviation / window
	devPerSqrt = math.sqrt(devPer)
	deviation = deviations * devPerSqrt
	for i in range(window):
		collrc_up[-(i+1)] = (linearRegression + slope * i) + deviation
		collrc_down[-(i+1)] = (linearRegression + slope * i) - deviation
	dtLRC['lrc_up'] = collrc_up.tolist()
	dtLRC['lrc_down'] = collrc_down.tolist()
	#return dtLRC[['lrc_up','lrc_down']]
	return dtLRC['lrc_up'],dtLRC['lrc_down'],intercept,slope

def SSLChannels(dataframe, length=7):
	df = dataframe.copy()
	df['ATR'] = ta.ATR(df, timeperiod=14)
	df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
	df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
	df['hlv'] = np.where(df['close'] > df['smaHigh'], 1,
	np.where(df['close'] < df['smaLow'], -1, np.NAN))
	df['hlv'] = df['hlv'].ffill()
	df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
	df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
	return df['sslDown'], df['sslUp']

def NadarayaWatsonEstimator2(dtloc, source = 'close', bandwidth = 8, window = 500):
	dtNWE = dtloc.copy()
	dtNWE['nwe_y1'] = np.nan
	dtNWE['nwe_y2'] = np.nan
	wn = np.zeros((window, window))
	for i in range(window):
		for j in range(window):
			wn[i,j] = math.exp(-(math.pow(i-j,2)/(bandwidth*bandwidth*2)))
	sumSCW = wn.sum(axis = 1)
	def calc_nwa(dfr, init=0):
		global calc_nwa_src_value
		if init == 1:
			calc_nwa_src_value = list()
			return
		calc_nwa_src_value.append(dfr[source])
		y1 = 0.0
		y2 = 0.0
		d = 0.0
		y1_val = 0.0
		y1_d = 0.0
		nweresult = 0
		if len(calc_nwa_src_value) > window:
			calc_nwa_src_value.pop(0)
		if len(calc_nwa_src_value) >= window:
			src = np.array(calc_nwa_src_value)
			sumSC = src * wn
			sumSCS = sumSC.sum(axis = 1)
			y2 = sumSCS / sumSCW
			y2_val = y2[-1]
			d = y2_val - y1
			y1 = y2
			y1_d = d
		return y1_d
	calc_nwa(None, init=1)
	dtNWE['nwe_y2'] = dtNWE.apply(calc_nwa, axis = 1, result_type='expand')
	return dtNWE['nwe_y2']

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
		x = np.arange(window)
		gradient, _ = np.polynomial.Polynomial.fit(x, y, deg=1)
		score = r2_score(x, y)
		return gradient#, score

	df['ssf_filter'] = pta.ssf(what, length = ssf_length, poles = 3)
	df['ssf_poly'] = df['ssf_filter'].rolling(rolling_window).apply(gradientas, raw = False)
	df['stdev'] = df['ssf_poly'].rolling(rolling_window, min_periods = 1).apply(np.std) * mult
	df['ssf_poly_u'] = df['ssf_poly'] + df['stdev']
	df['ssf_poly_l'] = df['ssf_poly'] - df['stdev']

	return df

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
	What: 
		dataframe kuri naudojam
	Length: 
		smoothing periodas
	d: 
		koeficientas
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
	df = dataframe.copy().fillna(0)
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

def range_filter(dataframe, what, length, mult):

	wper = length * 2 - 1

	df = dataframe.copy().fillna(0)

	df['what'] = getattr(df, what)
	df['x'] = df['avgrng'] = df['smoothrng'] = df['trc'] = 0
	df['long_short'] = df['upward'] = df['downward'] = 0

	for i in range(len(df)):
		df['x'].iat[i] = df['what'].iat[i]
		df['trc'].iat[i] = abs(df['what'].iat[i] - df['what'].iat[i-1])
		df['avgrng'].iat[i] = df['trc'].ewm(span = length, adjust = False).mean().iat[i]
		df['smoothrng'].iat[i] = (df['avgrng'].ewm(span = wper, adjust = False).mean().iat[i]) * mult

		df['x'].iat[i] = np.where(
			(df['x'].iat[i] > df['x'].iat[i-1]), 
				np.where(
					(df['x'].iat[i] - df['smoothrng'].iat[i]) < df['x'].iat[i - 1], df['x'].iat[i-1], df['x'].iat[i] - df['smoothrng'].iat[i]), 
						np.where(
							(df['x'].iat[i] + df['smoothrng'].iat[i] > df['x'].iat[i-1]), df['x'].iat[i-1], df['x'].iat[i] + df['smoothrng'].iat[i]))

		df['upward'].iat[i] = np.where(
			df['x'].iat[i] > df['x'].iat[i-1], df['upward'].iat[i-1] + 1, 
				np.where(
					df['x'].iat[i] < df['x'].iat[i-1], 0, df['upward'].iat[i-1]))

		df['downward'].iat[i] = np.where(
			df['x'].iat[i] < df['x'].iat[i-1], df['downward'].iat[i-1] + 1, 
				np.where(
					df['x'].iat[i] > df['x'].iat[i-1], 0, df['downward'].iat[i-1]))

		df['long_short'].iat[i] = np.where(
			(df['what'].iat[i] > df['x'].iat[i]) & (df['upward'].iat[i] > 0), 1,
				np.where(
					(df['what'].iat[i] < df['x'].iat[i]) & (df['downward'].iat[i] > 0), -1, 0
				))

	return df['x'], df['long_short']

def pinbar(df: DataFrame, smi=None):
	""" 
	Pinbar - Price Action Indicator
	
	Pinbars are an easy but sure indication
	of incoming price reversal. 
	Signal confirmation with SMI.
	
	Pinescript Source by PeterO
	https://tradingview.com/script/aSJnbGnI-PivotPoints-with-Momentum-confirmation-by-PeterO/
	
	:return: DataFrame with buy / sell signals columns populated
	"""
	
	low = df['low']
	high = df['high']
	close = df['close']
	
	tr = true_range(df)
	
	if smi is None:
		df = smi_momentum(df)
		smi = df['smi']
	
	df['pinbar_sell'] = (
		(high < high.shift(1)) &
		(close < high - (tr * 2 / 3)) &
		(smi < smi.shift(1)) &
		(smi.shift(1) > 40) &
		(smi.shift(1) < smi.shift(2))
	)

	df['pinbar_buy'] = (
		(low > low.shift(1)) &
		(close > low + (tr * 2 / 3)) &
		(smi.shift(1) < -40) &
		(smi > smi.shift(1)) &
		(smi.shift(1) > smi.shift(2))
	)
	
	return df