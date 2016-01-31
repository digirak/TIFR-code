import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import cauchy
freqs=np.arange(1800,4000,4)
N=freqs.size
lp=cauchy(0).pdf(np.arange(-1,1,1./10))
nu_max=0
offset=20
#act_offset=np.random.randint(10,high=20)
for i in range(offset+lp.size+5,N-lp.size-5-offset):
	act_offset=np.random.randint(0,high=20)
	spec=np.zeros(N)
	spec[i-lp.size/2:i+lp.size/2]=1*lp
	spec[i-lp.size-5:i-5]=.8*lp
	pos=np.arange(i-lp.size-5,i-5,1)
	pos_off=pos-act_offset
	spec[pos_off]=spec[pos]
	spec[pos]=0
	act_offset=np.random.randint(0,high=20)
	spec[i+5:i+lp.size+5]=.8*lp
	pos=np.arange(i+5,i+lp.size+5,1)
	pos_off=pos+act_offset
	spec[pos_off]=spec[pos]
	spec[pos]=0
	rand_noise=np.random.random(N)*np.random.randint(1,high=3)
	df=pd.DataFrame({'Frequency':freqs,
			 'l0':spec,
			 'noise':rand_noise,})
	df.to_csv('/home/rakesh/Fake_Data/Offset_trips/Test/Spec_numax_%d.csv'%(freqs[i]))
	
			     
				

