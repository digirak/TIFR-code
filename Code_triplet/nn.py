def nn_param(params,input):
	from theano import tensor as T
	layers=len(params)
	for lnum in range(layers):
		if (lnum==0):
			p=T.nnet.sigmoid(T.dot(input,params[lnum][0][1])+params[lnum][1][1])
			y=T.nnet.sigmoid(T.dot(p,T.transpose(params[lnum][0][1]))+params[lnum][2][1])
			yval=y.eval()
		else:
			p=T.nnet.sigmoid(T.dot(yval,params[lnum][0][1])+params[lnum][1][1])
			y=T.nnet.sigmoid(T.dot(p,T.transpose(params[lnum][0][1]))+params[lnum][2][1])
			yval=y.eval()
	
	return yval




