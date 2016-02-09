def plot_nn_fake(df):
	import numpy as np
	import cPickle
	from nn import nn_param
	from matplotlib import pyplot as plt
	f=file("nnparams.sav")
	update=cPickle.load(f)
	sig=df.l0
	sig_noise=df.l0+df.noise
	sig/=np.max(sig)
	sig_noise/=np.max(sig_noise)
	yval=nn_param(update, sig_noise)
	plt.subplot(2,2,1)
	plt.plot(sig_noise)
	plt.xlabel('$\\nu$ $\mu$Hz')
	N=df.Frequency.size
	plt.xticks(np.arange(0,N,N/4),df.Frequency[np.arange(0,N,N/4)])
	plt.title('Noisy Signal')
	
	plt.subplot(2,2,2)
	plt.plot(yval)
	plt.xlabel('$\\nu$ $\mu$Hz')
	N=df.Frequency.size
	plt.xticks(np.arange(0,N,N/4),df.Frequency[np.arange(0,N,N/4)])
	plt.title('Decoder Output')

	plt.subplot(2,2,4)
	plt.plot(sig)
	plt.xlabel('$\\nu$ $\mu$Hz')
	N=df.Frequency.size
	plt.xticks(np.arange(0,N,N/4),df.Frequency[np.arange(0,N,N/4)])
	plt.title('Noiseless spectra')

	plt.subplots_adjust(hspace=0.34)

def plot_nn_real(df):
	import numpy as np
	import cPickle
	from nn import nn_param
	from matplotlib import pyplot as plt
	f=file("nnparams.sav")
	update=cPickle.load(f)
	sig=(df.l0+df.l1+df.l2)*df.GaussProf
	sig_noise=(df.l0+df.l1+df.l2)*df.GaussProf+df.noise
	sig/=np.max(sig)
	sig_noise/=np.max(sig_noise)
	yval=nn_param(update, sig_noise)
	plt.subplot(2,2,1)
	plt.plot(sig_noise)
	plt.xlabel('$\\nu$ $\mu$Hz')
	N=df.Frequency.size
	plt.xticks(np.arange(0,N,N/4),df.Frequency[np.arange(0,N,N/4)])
	plt.title('Noisy Signal')
	
	plt.subplot(2,2,2)
	plt.plot(yval)
	plt.xlabel('$\\nu$ $\mu$Hz')
	N=df.Frequency.size
	plt.xticks(np.arange(0,N,N/4),df.Frequency[np.arange(0,N,N/4)])
	plt.title('Decoder Output')

	plt.subplot(2,2,4)
	plt.plot(sig)
	plt.xlabel('$\\nu$ $\mu$Hz')
	N=df.Frequency.size
	plt.xticks(np.arange(0,N,N/4),df.Frequency[np.arange(0,N,N/4)])
	plt.title('Noiseless spectra')

	plt.subplots_adjust(hspace=0.34)







