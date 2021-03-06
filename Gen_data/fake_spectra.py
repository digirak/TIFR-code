def fake_spec(nu_min,nu_max,df,delt_nu,numax,modes):
	from scipy.stats import norm
	from scipy.stats import cauchy
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	max_freq=nu_max
	min_freq=nu_min
	span=nu_max-nu_min
	N=int(span/df)+1
	Freq_axis=np.arange(min_freq,max_freq+1,df)
	freq_bins=np.arange(0,120,1)#Freq bins from 20 to 100 \muHz in steps of 1
	lmodes=2
	D0=1.5
	pos1=np.zeros(lmodes+1)
	epsilon=0
	spec=np.zeros(N)##l=0 spectra
	spec2=np.zeros(N)##l=2
	spec1=np.zeros(N)##l=1
        gmean=((((numax)-(np.mean(Freq_axis)))/df)/Freq_axis.size)
        gp=norm(gmean,0.25*2).pdf(np.arange(-1,1,2./N))#Gaussian profile for envelope
        lp=cauchy(3.0).pdf(np.arange(-2,2,2./60))#lorentzian profile for monopole
        lp2=cauchy(3.0).pdf(np.arange(-2,2,2./60))#lorentzian profile for dipole and quadrapole moments
	modes.size
	for i in range(1,modes+1):
	    if(i==1):
		spec[spec.size/2]=1.0
		spec2[spec.size/(2)+100]=0.5
		spec1[spec.size/(2)-100]=0.5
	    if(i>=1):
		spec[spec.size/(2**i)+epsilon]=1

		spec2[spec.size/(2**i)+4*lp.size-epsilon]=0.5
		spec2[spec.size/(2**i)+pos1[2]]=0.5

	 
		spec1[spec.size/(2**i)-4*lp.size+epsilon]=0.5
		#spec1[spec.size/(2**i)-pos1[1]]=0.5
		spec[spec.size-spec.size/(2**i)]=1
		spec2[spec.size-(spec.size/(2**i))+epsilon]=0.5
		#spec2[spec.size-(spec.size/(2**i)-pos1[2])]=0.5
		spec1[spec.size-(spec.size/(2**i)+4*lp.size)-epsilon]=0.5
		spec2[spec.size-(spec.size/(2**i)+pos1[1])]=0.5
	    """
	    Now add the lorentzian profile
	    """
	    for i in range(N):
		if(spec[i]==1.):
		    """
		    Check for all values equal to monopole
		    TODO:Find non-zero value 
		    """
		    spec[i]=0;
		#spec[i-lp.size/2:i+lp.size/2]+=lp*np.random.randint(1,high=10)#random amplitude variations
		    spec[i-lp.size/2:i+lp.size/2]+=lp
		
	    for i in range(N):
		if(spec2[i]==0.5):
		    """
		    Check for dipole/quadrapole
		    """
		    spec2[i]=0;
		#spec2[i-lp2.size/2:i+lp2.size/2]+=lp2*np.random.randint(1,high=10)#random amplitude variations
		    spec2[i-lp2.size/2:i+lp2.size/2]+=lp2

	    for i in range(N):
		if(spec1[i]==0.5):
		    spec1[i]=0;
		    #spec1[i-lp2.size/2:i+lp2.size/2]+=lp2*np.random.randint(1,high=10)#random amplitude variations
		    spec1[i-lp2.size/2:i+lp2.size/2]+=lp2
	    """
	    random noise generated with normal distribution
	    """
	    rand_noise=np.random.random(N)*np.random.randint(1,high=3)
	    print np.size(rand_noise)
	    spec_comb=spec+spec1+spec2
	    dat_frame=pd.DataFrame({'Frequency':Freq_axis[0:N],
				    'l0':spec,
				    'l1':spec1,
				    'l2':spec2,
				    'noise':rand_noise,})
	    dat_frame.to_csv('/home/rakesh/Fake_Data/Spec_numax_%d_modes_%d.csv'%(numax, modes))
  
