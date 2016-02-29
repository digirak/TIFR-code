from test_sdA import *
from stacked_dA import Stacked_dA
import theano.tensor as T
import numpy
from theano import function,pp
import timeit
import os
import sys

def test_dA(learning_rate, training_epochs,
            sig,sig_noise,chunks,batch_size=20,n_ins=441,n_hidden=1000):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
   # datasets = load_data(training_dataset,validation_dataset)

    pulsations= sig
    observations=sig_noise

    # compute number of minibatches for training, validation and testing
    n_train_batches = pulsations.get_value(borrow=True).shape[0] / batch_size
    print("Size of batch is %d"%batch_size)
    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('pulsations')  # the data is presented as rasterized images
    y = T.matrix('Observations')  # Noisy 
    # end-snippet-2

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)

    da = Stacked_dA(
        x,
        rng,
        n_visible=n_ins,

        n_hidden=n_hidden
    )
    cost, updates = da.get_cost_updates(corrupted_input=observations[index*batch_size:(index+1)*batch_size,:],learning_rate=learning_rate)
    train_da = function(
        inputs=[index],
      	outputs=[cost],
        updates=updates,
        givens={
                #y: observations[index * batch_size: (index + 1) * batch_size,:],
                x: pulsations[index * batch_size: (index + 1) * batch_size,:]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############
    n_train_batches=chunks

    # go through training epochs
    cos=[]
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(int(n_train_batches/batch_size)):
           # st_time=timeit.default_timer()
             c.append(train_da(batch_index))
            #end_time=timeit.default_timer()
            #print("training for batch %d/%d takes %3.3f s"%(batch_index,n_train_batches/batch_size,(end_time-st_time)))
	cos.append(numpy.mean([item for item in c]))

        print 'Training epoch %d ' % epoch

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %3.2f s' % ((training_time)))

    return (cos,updates)
#if __name__ == '__main__':
 #   test_dA()
