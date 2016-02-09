"""
This code will generate spectra using make_spec.
It will pass the parameters, numax and modes.
"""
import numpy as np
import matplotlib.pyplot as plt
from fake_spectra import fake_spec

df=1#microHz resolution
Freq_axis=np.arange(1800,4001,df)##microHz
modes=np.arange(1,4,1)
peak_freq=np.arange(1800,2100,10)
for i in range(peak_freq.size):#this is numax loop
	for j in range(modes.size):
		fake_spec(Freq_axis[0],Freq_axis[Freq_axis.size-2],df,80.1,peak_freq[i],modes[j]);
		print(peak_freq[i])





