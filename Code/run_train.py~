import train_spectra
from train_spectra import train_spectra
import numpy as np
import pandas as pd
import cPickle as cp
#import os
#procno=int(os.environ["PBS_VNODENAME"])
num_modes=1
min_modes=12
#cost=np.zeros(num_modes*nu_max.size)
#cost=cost.reshape(num_modes,nu_max.size)
import theano
nhidden=1000
learn_rate=1e-1
updts_old=0.
epoch=0.
updts_new=0.
epochs=20
cost=np.zeros(epochs)
for k in range(epochs):
	modes=np.random.uniform(12,25,num_modes)
	nu_max=np.asarray([2000,2200,2500,2700,3000,3300,3500])
	np.random.shuffle(nu_max)
	for j in range(nu_max.size):
		for i in range(num_modes):
			if (updts_old==0.):
				[c,updts_old]=train_spectra(nu_max[j],modes[i],nhidden,learn_rate,W=0,bhid=0,bvis=0)
				#cost[i-min_modes,j]=c.eval()
				updts_old=[[theano.shared(a.eval()),theano.shared(b.eval())] for (a,b) in updts_old]
			else:
				W=updts_old[0][1]
				bhid=updts_old[1][1]
				bvis=updts_old[2][1]
				[c,updts_new]=train_spectra(nu_max[j],modes[i],nhidden,learn_rate,W=W,bhid=bhid,bvis=bvis)
				#cost[i-min_modes,j]=c.eval()
				updts_old=[[theano.shared(a.eval()),theano.shared(b.eval())] for (a,b) in updts_new]
			cost[k]+=c.eval()	
	
	print("completed training in epoch %d" %(k))
df=pd.DataFrame(cost)
df.to_csv('/home/rakesh/sandbox_Code/Code/Results/cost.csv')
f=file('/home/rakesh/sandbox_Code/Code/Results/nnparams.sav','wb')
cp.dump(updts_old,f,protocol=cp.HIGHEST_PROTOCOL)
f.close()



