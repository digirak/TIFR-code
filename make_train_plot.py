import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/rakesh/Rak_lib")
from nn import nn_param
from plot_nn import just_plot
import pandas as pd
import cPickle

sig=np.load("/home/rakesh/Data/CV_twins/sig.npy")
sig_noise=np.load("/home/rakesh/Data/CV_twins/sig_noise.npy")
df=pd.read_csv('/home/rakesh/Fake_Data/MultiTrip/Spec_numax_1956.csv')
nu=np.asarray(df.Frequency)
nu_size=nu.size
f=file("/home/rakesh/Code/Code_gpuopt/Results/nnparams.sav")
nn=cPickle.load(f)
for i in range(150):
    yval=nn_param(nn,sig_noise[i,:])
    just_plot(sig_noise[i,:],yval,sig[i,:],df)
    plt.savefig("/home/rakesh/Plots/08Feb/CV/numax_%d.png"%(nu[i]))
    plt.close()


