import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from mlp import HiddenLayer
from dA_class import dA

class Stacked_dA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
	input,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
	W=None,
	bhid=None,
	bvis=None,
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

"""

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
	self.x=input
	self.bvis=bvis

        assert self.n_layers > 0

	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
	
            if W is None and bvis is None and bhid is None:
		    sigmoid_layer = HiddenLayer(rng=numpy_rng,
						input=input,
						n_in=input.size,
						n_out=hidden_layers_sizes[i],
						activation=T.nnet.sigmoid)
	    else:
		    sigmoid_layer = HiddenLayer(rng=numpy_rng,
						input=input,
						n_in=input.size,
						n_out=hidden_layers_sizes[i],
						W=W[i],
						b=bhid[i],
						activation=T.nnet.sigmoid)
		    
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
	    if bvis is None:
		    dA_layer = dA(numpy_rng=numpy_rng,
				  input=input,
				  n_visible=input.size,
				  n_hidden=hidden_layers_sizes[i],
				  W=sigmoid_layer.W,
				  bhid=sigmoid_layer.b,
				  bvis=None)
	    else:
		    dA_layer = dA(numpy_rng=numpy_rng,
				  input=input,
				  n_visible=input.size,
				  n_hidden=hidden_layers_sizes[i],
				  W=sigmoid_layer.W,
				  bhid=sigmoid_layer.b,
				  bvis=self.bvis[i])

            self.dA_layers.append(dA_layer)

	    
