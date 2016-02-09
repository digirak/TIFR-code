def squared_error(yval,sig):
	from lasagne.objectives import squared_error
	return squared_error(yval,sig)
def entropy_error(yval,sig):
	from lasagne.objectives import categorical_crossentropy
	return categorical_crossentropy(yval,sig)
def compute_cost_fake(df):
	"""
	Pass a pandas dataframe df and then generate signal.
	Call the entropy error
	and return the averaged cost
	"""
	import numpy as np
	import cPickle
	from nn import nn_param
	from matplotlib import pyplot as plt
	from theano import tensor as T
	f=file("nnparams.sav")
	update=cPickle.load(f)
	sig=np.asarray(df.l0)
	sig_noise=np.asarray(df.l0+df.noise)
	sig/=np.max(sig)
	sig_noise/=np.max(sig_noise)
	yval=nn_param(update,sig_noise)
	return T.mean(squared_error(yval,sig))
def compute_cost_real(df):
	"""
	Pass a pandas dataframe df and then generate signal.
	Call the entropy error
	and return the averaged cost
	"""
	import numpy as np
	import cPickle
	from nn import nn_param
	from matplotlib import pyplot as plt
	from theano import tensor as T
	f=file("nnparams.sav")
	update=cPickle.load(f)
	sig=(df.l0+df.l1+df.l2)*df.GaussProf
	sig_noise=sig+df.noise
	sig/=np.max(sig)
	sig_noise/=np.max(sig_noise)
	yval=nn_param(update,sig_noise)
	return T.mean(entropy_error(yval,sig))
