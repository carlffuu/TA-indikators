import numpy as np
import scipy
import talib.abstract as ta
import math
import pandas as pd
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib

def renko(dataframe, what, ha = True, atr = 1, mult = 0.5):
	if ha == True:
		heikinashi = qtpylib.heikinashi(dataframe)
		dataframe['ATR'] = ta.ATR(heikinashi, timeperiod = atr).fillna(1)
	else:
		dataframe['ATR'] = ta.ATR(dataframe, timeperiod = atr).fillna(1)
	renko_columns = [
		'date', 
		'renko_open', 
		'renko_high', 
		'renko_low', 
		'renko_close', 
		'trend', 
		'prev_trend', 
		'prev2_trend', 
		'prev3_trend', 
		'prev_date', 
		'new_brick', 
	]
	DATE_IDX = 0
	CLOSE_IDX = 4
	TREND_IDX = 5
	PREV_TREND_IDX = 6
	PREV2_TREND_IDX = 7
	NEW_BRICK = True
	COPIED_BRICK = False

	brick_size = np.NaN

	data = []
	prev_brick = None
	prev2_trend = False
	prev3_trend = False
	for row in dataframe.itertuples():
		if np.isnan(row.ATR):
			continue
		else:
			brick_size = row.ATR * mult

		close = getattr(row, what)
		#close = what
		date = row.date

		if prev_brick is None:
			trend = True
			prev_brick = [
				date, 
				close - brick_size, 
				close, 
				close - brick_size, 
				close, 
				trend, 
				False, 
				False, 
				False, 
				date, 
				NEW_BRICK, 
			]
			prev2_trend = prev_brick[PREV_TREND_IDX]
			prev3_trend = prev_brick[PREV2_TREND_IDX]
			data.append(prev_brick)
			continue

		prev_date = prev_brick[DATE_IDX]
		prev_close = prev_brick[CLOSE_IDX]
		prev_trend = prev_brick[TREND_IDX]

		new_brick = None
		trend = prev_trend

		bricks = int(np.nan_to_num((close - prev_close) / brick_size))

		if trend and bricks >= 1:
			new_brick = [
				date, 
				prev_close, 
				prev_close + bricks * brick_size, 
				prev_close, 
				prev_close + bricks * brick_size, 
				trend, 
				prev_trend, 
				prev2_trend, 
				prev3_trend, 
				prev_date, 
				NEW_BRICK, 
			]

		elif trend and bricks <= -2:
			trend = not trend

			new_brick = [
				date, 
				prev_close - brick_size, 
				prev_close - brick_size, 
				prev_close - abs(bricks) * brick_size, 
				prev_close - abs(bricks) * brick_size, 
				trend, 
				prev_trend, 
				prev2_trend, 
				prev3_trend, 
				prev_date, 
				NEW_BRICK, 
			]

		elif not trend and bricks <= -1:
			new_brick = [
				date, 
				prev_close, 
				prev_close, 
				prev_close - abs(bricks) * brick_size, 
				prev_close - abs(bricks) * brick_size, 
				trend, 
				prev_trend, 
				prev2_trend, 
				prev3_trend, 
				prev_date, 
				NEW_BRICK, 
			]

		elif not trend and bricks >= 2:
			trend = not trend

			new_brick = [
				date, 
				prev_close + brick_size, 
				prev_close + bricks * brick_size, 
				prev_close + brick_size, 
				prev_close + bricks * brick_size, 
				trend, 
				prev_trend, 
				prev2_trend, 
				prev3_trend, 
				prev_date, 
				NEW_BRICK, 
			]

		else:
			data.append(
				[
					date, 
					prev_brick[1], 
					prev_brick[2], 
					prev_brick[3], 
					prev_brick[4], 
					prev_brick[5], 
					prev_brick[6], 
					prev_brick[7], 
					prev_brick[8], 
					prev_brick[9], 
					COPIED_BRICK, 
				]
			)

		if new_brick is not None:
			data.append(new_brick)
			prev2_trend = prev_brick[PREV_TREND_IDX]
			prev3_trend = prev_brick[PREV2_TREND_IDX]
			prev_brick = new_brick
	renko_chart = pd.DataFrame(data = data, columns = renko_columns)
	return renko_chart['renko_open'], renko_chart['renko_close']