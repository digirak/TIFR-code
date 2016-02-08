#@profile
def train_spectra(sig,sig_noise,sig_size,nu_max,modes,nhidden,learn_rate,num_layers,W=0,bhid=0,bvis=0):
	import pandas as pd
	import sys
        import gc
	sys.path.append("/home/rakesh/Gen_data/")
	import stacked_dA
	from stacked_dA import Stacked_dA
	import numpy as np
	from lasagne.updates import apply_momentum
	import theano.tensor as T
	import theano
	import make_multtip
	from make_multtip import make_sing_multtrip
	import timeit
	#pathname="/home/rakesh/Fake_Data/MultiTrip/Spec_numax_%d.csv"%(nu_max)
	#df=pd.read_csv(pathname)
	#sig=np.asarray((df.l0)*1)
	#sig/=np.max(sig)
	#sig_noise=np.asarray((df.l0)*1+df.noise)
	#sig_noise/=np.max(sig)
#	[sig,sig_noise]=make_sing_multtrip(nu_max)
	numrand=np.random.RandomState(123)
	hidd_layers=[nhidden]*num_layers
	if W is 0 and bhid is 0 and bvis is 0:
		stack_enc=Stacked_dA(sig,numrand,n_ins=sig_size,hidden_layers_sizes=hidd_layers,n_outs=sig_size)
	else:
		stack_enc=Stacked_dA(sig,numrand,n_ins=sig_size,hidden_layers_sizes=hidd_layers,n_outs=sig_size,W=W,bhid=bhid,bvis=bvis)
	cost=[]
	update=[]
	#[cost,update]=stack_enc.get_cost_updates(sig_noise,learning_rate=learn_rate)
	for i in range(num_layers):
		if i==0:
			[cos,ups]=stack_enc.dA_layers[i].get_cost_updates(sig_noise,learn_rate)
			p=T.nnet.sigmoid(T.dot(sig_noise,ups[0][1])+ups[1][1])
			y=T.nnet.sigmoid(T.dot(p,T.transpose(ups[0][1]))+ups[2][1])
			#yval=y.eval()

		else:
			[cos,ups]=stack_enc.dA_layers[i].get_cost_updates(y,learn_rate)
			p=T.nnet.sigmoid(T.dot(y,ups[0][1])+ups[1][1])

			y=T.nnet.sigmoid(T.dot(p,T.transpose(ups[0][1]))+ups[2][1])
			#yval=y.eval()
		cost.append(cos)
		ups=[[theano.shared(a.eval()),theano.shared(b.eval())] for (a,b) in ups]
		update.append(ups)
        #gc.collect()
        del stack_enc
	return (cost,update)
