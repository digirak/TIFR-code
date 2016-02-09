def run_train(nnparams=0):
    import train_deep_spectra
    from train_deep_spectra import train_spectra
    import numpy as np
    import pandas as pd
    import cPickle as cp
    import timeit
    from theano import tensor as T
    #from theano import shared

    #import os
    #procno=int(os.environ["PBS_VNODENAME"])
    num_modes=3
    min_modes=12
    #cost=np.zeros(num_modes*nu_max.size)
    #cost=cost.reshape(num_modes,nu_max.size)
    import theano
    nhidden=3000
    learn_rate=3e-3
    updts_old=nnparams
    epoch=0.
    updts_new=0.
    epochs=8
    cost_perepoch=np.zeros(epochs)
    num_layers=1
    modes=np.arange(1,4)
    freqs=np.arange(1956,3000,4)
    sig_noise=np.load('Data/Train_twins/sig_noise.npy')
    sig=np.load('Data/Train_twins/sig.npy')
    sig_size=np.size(sig[0,:])
    sig_t=theano.shared(np.asarray(sig,dtype=theano.config.floatX),borrow=True)
    sig_noise_t=theano.shared(np.asarray(sig_noise,dtype=theano.config.floatX),borrow=True)
    for k in range(epochs):
     #np.random.shuffle(modes)
            cost=[]
            start_time=timeit.default_timer()
            nu_max=np.asarray(np.arange(1956,3000,4))
           # np.random.shuffle(nu_max)
            shufs=np.random.permutation(len(nu_max))
            for j in range(nu_max.size):
                    #for i in range(num_modes):
                   # p=np.ravel(np.where(freqs==nu_max[j]))
                    p=shufs[j]
                    start_sample_time=timeit.default_timer()
                    if (updts_old==0.):
                            #st=timeit.default_timer()
                            [c,updts_old]=train_spectra(sig_t[p,:],sig_noise_t[p,:],sig_size,nu_max[p],modes[0],nhidden,learn_rate,num_layers=num_layers,W=0,bhid=0,bvis=0)
                            #en=timeit.default_timer()
                            #print("takes %3.4f s in one train"%(en-st))
                                    #cost[i-min_modes,j]=c.eval()
                    else:
                            Wlist=[]
                            blist=[]
                            bvislist=[]
                            for item in range(len(updts_old)):
                                    Wlist.append(updts_old[item][0][1])
                                    blist.append(updts_old[item][1][1])
                                    bvislist.append(updts_old[item][2][1])
                            updts_old=0.
                            #st=timeit.default_timer()
                            [c,updts_old]=train_spectra(sig_t[p,:],sig_noise_t[p,:],sig_size,nu_max[p],modes[0],nhidden,learn_rate,num_layers=num_layers,W=Wlist,bhid=blist,bvis=bvislist)
                            #en=timeit.default_timer()
                            #print("takes %3.4f s in one train"%(en-st))
                            #cost[k]+=np.mean([item.eval() for item in c])
                            #cost[k]+=c
                    cost.append(c[0].eval())
                    end_sample_time=timeit.default_timer()
                    print("Sample %f takes %3.4f s time"%(j,(end_sample_time-start_sample_time)))
           # cost_perepoch[k]+=np.mean([item.eval() for item in cost])

            end_time=timeit.default_timer()
            #print(sum(cost))
            cost_perepoch[k]=np.mean(cost,axis=0,dtype=np.float32)
            print("Completed training in epoch %d in %3.4fs"%((k+1),(end_time-start_time)))

    return(cost_perepoch,updts_old)        
  #  df=pd.DataFrame(cost_perepoch/k)
   # df.to_csv('/home/rakesh/Code/Code_gpuopt/Results/cost.csv')
   # f=file('/home/rakesh/Code/Code_gpuopt/Results/nnparams.sav','wb')
    #cp.dump(updts_old,f,protocol=cp.HIGHEST_PROTOCOL)
   # f.close()



