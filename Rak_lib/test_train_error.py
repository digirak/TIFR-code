import sys
sys.path.append("/home/rakesh/Rak_lib")
import numpy as np
import pandas as pd
import compute_error
from compute_error import compute_cost_fake


pathname_test="/home/rakesh/Fake_Data/MultiTrip/Test/Spec_numax_"
pathname_train="/home/rakesh/Fake_Data/MultiTrip/Spec_numax_"
train_cost=[]
test_cost=[]
cost=[]
for nu in range(1980,3000,4):
	df=pd.read_csv(pathname_train+"%d.csv"%(nu))
	cost.append(compute_cost_fake(df))

temp_cost=[item.eval() for item in cost]
for item in range(len(cost)):
	train_cost.append(np.mean(temp_cost[0:item]))
cost=[]
for nu in range(1980,3000,4):
	df=pd.read_csv(pathname_test+"%d.csv"%(nu))
	cost.append(compute_cost_fake(df))
temp_cost=0.
temp_cost=[item.eval() for item in cost]
for item in range(len(cost)):
	test_cost.append(np.mean(temp_cost[0:item]))

df=pd.DataFrame({"Test":test_cost,
		"Train":train_cost,})
df.to_csv("/home/rakesh/Training_curve/curve.csv")


