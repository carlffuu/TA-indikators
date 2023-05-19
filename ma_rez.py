import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from technical.indicators import VIDYA

from indikatoriai import *
from monotonic import *
from tools import *

def rez_ma(close, high, low, volume, df, velo_length = 20, ma_list = [], ma_type = 'ssf', thresh = 0, mono_sens = 0, pref = ''):
	data = []
	dfcopy = df.copy()
	df = pd.DataFrame(index = dfcopy.index)

	def ma_velo(df, velo_length = velo_length, k = 1):
		df_max = df.rolling(velo_length).max()
		df_min = df.rolling(velo_length).min()
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
	df[f'{ma_type}_{pref}-vel'] = ma_velo(df[f'{ma_type}_{pref}'], velo_length = velo_length, k = 1)
	df[f'{ma_type}_{pref}-velo_trend'] = np.where(
		(df[f'{ma_type}_{pref}-vel'] == 1), 1, 
		np.where(
		(df[f'{ma_type}_{pref}-vel'] == 0), -1, 0))

	df[f'{ma_type}_{pref}-velo_dir'] = np.where(df[f'{ma_type}_{pref}-vel'] > df[f'{ma_type}_{pref}-vel'].shift(1), 1, -1)
	df[f'{ma_type}_{pref}-velo_dir+'] = np.where(
		(df[f'{ma_type}_{pref}-vel'] > 0.95) | (df[f'{ma_type}_{pref}-velo_dir'] == 1), 1, 
		(np.where((df[f'{ma_type}_{pref}-vel'] < 0.05) | (df[f'{ma_type}_{pref}-velo_dir'] == -1), -1, 0)))

	#### Angle
	df[f'{ma_type}_{pref}-angle'] = ta.LINEARREG_ANGLE(df[f'{ma_type}_{pref}'])
	df[f'{ma_type}_{pref}-angle_trend'] = np.where((df[f'{ma_type}_{pref}-angle'] > 0), 1, -1)

	#### Mono
	df[f'{ma_type}_{pref}-mono_line'], df[f'{ma_type}_{pref}-mono_thr_long'], df[f'{ma_type}_{pref}-mono_thr_short'], df[f'{ma_type}_{pref}-mono_dir_up'], df[f'{ma_type}_{pref}-mono_dir_dn'] = monotonic_advanced(what = df[f'{ma_type}_{pref}'], length = 6, thresh_p = thresh, thresh_n = 100-thresh, window = 700, sens = mono_sens)

	df[f'{ma_type}_{pref}-mono_2_dir_up'], df[f'{ma_type}_{pref}-mono_2_dir_dn'] = monotonic_simple(what = df[f'{ma_type}_{pref}'], sens = mono_sens, mode = 1)

	df[f'{ma_type}_{pref}-angle_mono_dir_up'], df[f'{ma_type}_{pref}-angle_mono_dir_dn'] = monotonic_simple(what = df[f'{ma_type}_{pref}-angle'], sens = 0.12, mode = 1)

	#### Correl
	df[f'{ma_type}_{pref}-correlo_close'] = ta.CORREL(df[f'{ma_type}_{pref}'], close, timeperiod = 20)
	mean_volume = (meannormalize(volume, window = 150).fillna(0)*2)
	df[f'{ma_type}_{pref}-correlo_vol'] = ta.CORREL(df[f'{ma_type}_{pref}'], mean_volume, timeperiod = 20)
	
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
