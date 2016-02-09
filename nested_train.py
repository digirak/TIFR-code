import cPickle as cp
import run_chunked_train
import os
from run_chunked_train import run_train
import timeit
import numpy as np
#import pandas as pd
#path="/home/rakesh/Code/Code_gpuopt/Results/nnparams.sav"
#if(os.path.isfile(path)):
 #   os.remove(path)
epochs=5
nnparams=0.
cost=[]
for epoch in range(epochs):
#   if(os.path.isfile(path)):
 #       f=file(path)
  #      nnparams=cPickle.load(f)
   # else:
    st_time=timeit.default_timer() 
    [c,nnparams]=run_train(nnparams=nnparams)
    en_time=timeit.default_timer()
    print("Epoch number %d took %3.4f s"%(8*(epoch+1),(en_time-st_time)))
    cost.append(c)
    c=0
print("completed training check Results")
#df=pd.DataFrame(cost)
#df.to_csv('/home/rakesh/Code/Code_gpuopt/Results/cost.csv')
np.save("/home/rakesh/Code/Code_gpuopt/Results/cost.npy",cost)
f=file('/home/rakesh/Code/Code_gpuopt/Results/nnparams.sav','wb')
cp.dump(nnparams,f,protocol=cp.HIGHEST_PROTOCOL)
f.close()
