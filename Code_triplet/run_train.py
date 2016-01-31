import train_deep_spectra
from train_deep_spectra import train_spectra
import numpy as np
import pandas as pd
import cPickle as cp
import timeit
#import os
#procno=int(os.environ["PBS_VNODENAME"])
num_modes=3
min_modes=12
#cost=np.zeros(num_modes*nu_max.size)
#cost=cost.reshape(num_modes,nu_max.size)
import theano
nhidden=1000
learn_rate=3e-2
updts_old=0.
epoch=0.
updts_new=0.
epochs=25
cost=np.zeros(epochs)
num_layers=1
modes=np.arange(1,4)
for k in range(epochs):
 #np.random.shuffle(modes)
	nu_max=np.asarray(np.arange(1900,2501,4))
	np.random.shuffle(nu_max)
	for j in range(nu_max.size):
		start_time=timeit.default_timer()
		if (updts_old==0.):
			[c,updts_old]=train_spectra(nu_max[j],modes[0],nhidden,learn_rate,num_layers=num_layers,W=0,bhid=0,bvis=0)
				#cost[i-min_modes,j]=c.eval()
		else:
			Wlist=[]
			blist=[]
			bvislist=[]
			for item in range(len(updts_old)):
				Wlist.append(updts_old[item][0][1])
				blist.append(updts_old[item][1][1])
				bvislist.append(updts_old[item][2][1])
			updts_old=0.
			[c,updts_old]=train_spectra(nu_max[j],modes[0],nhidden,learn_rate,num_layers=num_layers,W=Wlist,bhid=blist,bvis=bvislist)
				#cost[i-min_modes,j]=c.eval()
		end_time=timeit.default_timer()
		print("Sample %d completed in %d s time"%(j,(end_time-start_time)))
		cost[k]+=np.mean([item.eval() for item in c])
	
print("completed training in epoch %d" %(k))
df=pd.DataFrame(cost/k)
df.to_csv('/home/rakesh/sandbox_Code/Code_triplet/Results/cost.csv')
f=file('/home/rakesh/sandbox_Code/Code_triplet/Results/nnparams.sav','wb')
cp.dump(updts_old,f,protocol=cp.HIGHEST_PROTOCOL)
f.close()



