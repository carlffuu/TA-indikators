import numpy as np
import scipy
import talib.abstract as ta
import math
import pandas as pd
import pandas_ta as pta

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

def CDV(self, dataframe):
	df = dataframe.copy().fillna(0)

	df['tw'] = 0
	df['bw'] = 0
	df['body'] = 0

	df['rate_cond_1'] = 0
	df['rate_cond_2'] = 0
	df['rate_1'] = 0
	df['rate_2'] = 0

	df['deltaup'] = 0
	df['deltadown'] = 0

	df['delta'] = 0
	df['cdv'] = 0

	last_row = df.tail(1).index.item()
	for i in range(self.startup_candle_count, last_row + 1):

		df['tw'].iat[i] = df['high'].iat[i] - max(df['open'].iat[i], df['close'].iat[i])
		df['bw'].iat[i] = min(df['open'].iat[i], df['close'].iat[i]) - df['low'].iat[i]
		df['body'].iat[i] = abs(df['close'].iat[i] - df['open'].iat[i])

		df['rate_cond_1'].iat[i] = 0.5 * (df['tw'].iat[i] + df['bw'].iat[i] + (np.where(df['open'].iat[i] <= df['close'].iat[i], 2 * df['body'].iat[i], 0))) / (df['tw'].iat[i] + df['bw'].iat[i] + df['body'].iat[i])
		df['rate_cond_2'].iat[i] = 0.5 * (df['tw'].iat[i] + df['bw'].iat[i] + (np.where(df['open'].iat[i] > df['close'].iat[i], 2 * df['body'].iat[i], 0))) / (df['tw'].iat[i] + df['bw'].iat[i] + df['body'].iat[i])
		df['rate_1'].iat[i] = np.where(df['rate_cond_1'].iat[i] == 0, 0.5, df['rate_cond_1'].iat[i])
		df['rate_2'].iat[i] = np.where(df['rate_cond_2'].iat[i] == 0, 0.5, df['rate_cond_2'].iat[i])

		df['deltaup'].iat[i] = df['volume'].iat[i] * df['rate_1'].iat[i]
		df['deltadown'].iat[i] = df['volume'].iat[i] * df['rate_2'].iat[i]

		df['delta'].iat[i] = np.where(df['close'].iat[i] >= df['open'].iat[i], df['deltaup'].iat[i], -df['deltadown'].iat[i])
		df['cdv'].iat[i] = df['cdv'].iat[i-1] + df['delta'].iat[i]

	#df['cdv'] = df['delta'].rolling(self.startup_candle_count).sum()
	return df['cdv'], df['delta']

def volume_osc(dataframe, fast_length, slow_length):
	fast = ta.EMA(dataframe['volume'], timeperiod = fast_length)
	slow = ta.EMA(dataframe['volume'], timeperiod = slow_length)
	osc = 100 * (fast - slow) / slow
	osc_p = np.where(osc >= 0, osc, np.nan)
	osc_n = np.where(osc <= 0, osc, np.nan)
	return osc_p, osc_n