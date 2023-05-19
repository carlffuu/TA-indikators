import numpy as np
import scipy
import talib.abstract as ta
import math
import pandas as pd
import pandas_ta as pta
from tools import *

def monotonic_simple(what, sens, mode):
	"""
	Monotonic paprastas, jei a > b nepriklausomai
	What:
		dataframe kuri naudojam
	Mode:
		1 = returnina up ir down
		2 = returnina viena reiksme
	Sens:
		jautrumas, nustatom pagal save
	"""
	a = np.where((what > what.shift(1)) & (abs(what - what.shift(1)) > sens), 1, 0) 	#mono dir up
	b = np.where((what < what.shift(1)) & (abs(what - what.shift(1)) > sens), -1, 0) 	#mono dir dn
	if (mode == 1):
		return a, b
	if (mode == 2):
		c = np.where(a == 1, 1, np.where(b == -1, -1 ,0))
		return c

def monotonic_change(self, dataframe, what, sens, mode):
	"""
	Monotonic procentinis change kuris naudoja truechange funkcijas.
	What:
		dataframe kuri naudojam
	Mode:
		1 = returnina up ir down
		2 = returnina viena reiksme (1, -1, 0)
	Sens:
		jautrumas, nustatom pagal save
	"""
	z = truechange(self, dataframe, what, mode = 1)
	#a = np.where((what > what.shift(1)) & (abs(truechange(what) - truechange(what.shift(1))) > sens), 1, 0) 	#mono dir up
	#b = np.where((what < what.shift(1)) & (abs(truechange(what) - truechange(what.shift(1))) > sens), -1, 0) 	#mono dir dn
	#a = np.where((what > what.shift(1)) & (z > z.shift(1)) & (abs(z - z.shift(1)) > sens), 1, 0) 	#mono dir up
	#b = np.where((what < what.shift(1)) & (z > z.shift(1)) & (abs(z - z.shift(1)) > sens), -1, 0) 	#mono dir dn
	a = np.where((what > what.shift(1)) & (abs(z - z.shift(1)) > sens), 1, 0) 	#mono dir up
	b = np.where((what < what.shift(1)) & (abs(z - z.shift(1)) > sens), -1, 0) 	#mono dir dn
	if (mode == 1):
		return a, b
	if (mode == 2):
		c = np.where((a == 1) & (b > -1), 1, np.where((b == -1) & (a < 1), -1 ,0))
		return c

def monotonic_advanced(what, length, thresh_p, window, thresh_n, sens):
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