import numpy as np
from sigmoid import sigmoid
def aeneunet(input,num_layers,num_weights):
	num_biases=num_layers
	x=sigmoid(input)
	x=np.append(1,x)
	theta=np.zeros(num_weights+1)##intialize all weights to zeros
	L1=(x).dot(np.transpose(theta))
	a1=np.sigmoid(L1)
	return a1






