import pandas as pd
import plot_nn
from plot_nn import plot_nn_fake
import numpy as np 

nu_max=np.arange(1980,3000,4)
for nu in range(nu_max.size):
	path="/home/rakesh/Fake_data/Offset_trips/Spec_numax_%d"%nu_max[nu]
	df=pd.read_csv(path)
	plot_nn_fake(df)
	plt.savefig("/home/Rakesh/Plots/26Jan/Train/numax_%d"%nu_max[nu])
	plt.close()



