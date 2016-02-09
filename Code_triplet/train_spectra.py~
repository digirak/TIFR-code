def train_spectra(nu_max,modes,nhidden,learn_rate,W=0,bhid=0,bvis=0):
	import pandas as pd
	import dA_class
	from dA_class import dA
	import numpy as np
	from lasagne.updates import apply_momentum
	pathname="/home/rakesh/Fake_Data/Spec_numax_%d_modes_%d.csv"%(nu_max,modes)
	df=pd.read_csv(pathname)
	sig=np.asarray((df.l0+df.l1+df.l2)*df.GaussProf)
	sig/=np.max(sig)
	sig_noise=np.asarray((df.l0+df.l1+df.l2)*df.GaussProf+df.noise)
	sig_noise/=np.max(sig)
	randnum=np.random.RandomState(123)
	if W is 0 and bhid is 0 and bvis is 0:
		enc=dA(numpy_rng=randnum,input=sig,n_visible=sig.size,n_hidden=nhidden)
	else:
		enc=dA(numpy_rng=randnum,input=sig,n_visible=sig.size,n_hidden=nhidden,W=W,bhid=bhid,bvis=bvis)
	
	[cost,update]=enc.get_cost_updates(sig_noise,learning_rate=learn_rate)




	return (cost,update)








	

	
