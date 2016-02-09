def make_spec(nu_min,nu_max,df,delt_nu,numax,modes):
    from scipy.stats import norm
    from scipy.stats import cauchy
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    max_freq=nu_max
    min_freq=nu_min
    span=nu_max-nu_min
    N=int(span/df)+1
    Freq_axis=np.arange(min_freq,max_freq,df)
    freq_bins=np.arange(0,120,1)#Freq bins from 20 to 100 \muHz in steps of 1

    lmodes=2
    D0=1.5
    pos1=np.zeros(lmodes+1)

    spec=np.zeros(N)##l=0 spectra
    spec2=np.zeros(N)##l=2
    spec1=np.zeros(N)##l=1
    """
    Generate both gaussian and lorentzian profile
    """
    gmean=((((numax)-(np.mean(Freq_axis)))/df)/Freq_axis.size)
    gp=norm(gmean,0.25*2).pdf(np.arange(-1,1,2./N))#Gaussian profile for envelope
    """
    Size of lorentzian is arbitrary
    """
    lp=cauchy(0.5).pdf(np.arange(-2,2,2./60))#lorentzian profile for monopole
    lp2=cauchy(0.5).pdf(np.arange(-2,2,2./60))#lorentzian profile for dipole and quadrapole moments
    """
    Generate spectral positions
    30/12/15: Realized the frequency separations have to be included to make the modes 'solar like'
    Arbitrary amplitude now, radial/monopole mode has amplitude of 1 and dipole/quadripole
have amplitude 0.5
    TODO:Change that the randome number so amplitudes are randomized.
    """
#epsilon=np.random.randint(1,high=50)
#epsilon=1.5#test epsilon
    epsilon=np.zeros(N)
    a=-4.73
    b=-2.0
    epsilon=a*(Freq_axis/3100)**b
    epsilon/=df
    """
    The idea is we now have n^2 number of spectra on the PS
    I will now define two positions, pos_1 corresponding to position of l=1 and pos_2 corresponding to position
of l=2.
    pos_1 is indexed as nx1 and pos_2 is a nx1 vector.
    """
    pos_0=np.zeros(modes+1) #positions of n=1,2,3 modes
    for i in range(1,modes+1):
        pos_0[i]=int((delt_nu*i/df))
        for i in range(1,lmodes+1):
            pos1[i]=int(((delt_nu*i/2)-((i*(i+1)*D0)))/df)##generate offset of l=1,2 modes
   # pos1[i]=int((((6*(i+4)*D0)))/df)##generate offset of l=1,2 modes
    for i in range(1,modes+1):
        spec[pos_0[i]+int(epsilon[pos_0[i]])]=1##adding epsilon component here
        spec1[pos_0[i]+pos1[1]+int(epsilon[pos_0[i]])]=0.5
        spec2[pos_0[i]+pos1[2]+int(epsilon[pos_0[i]])]=0.5
    """
    This segment is the initial code where I had 
    just put in spectra at random pre-set postions
    """
   # if(i>=1):
 #       spec[spec.size/(2**i)+epsilon]=1

 #       spec2[spec.size/(2**i)+4*lp.size-epsilon]=0.5
        #spec2[spec.size/(2**i)+pos1[2]]=0.5

 
 #       spec1[spec.size/(2**i)-4*lp.size+epsilon]=0.5
      #  spec1[spec.size/(2**i)-pos1[1]]=0.5
       # spec[spec.size-spec.size/(2**i)]=1
        #spec2[spec.size-(spec.size/(2**i)-4*lp.size)+epsilon]=0.5
       # spec2[spec.size-(spec.size/(2**i)-pos1[2])]=0.5
        #spec1[spec.size-(spec.size/(2**i)+4*lp.size)-epsilon]=0.5
       # spec1[spec.size-(spec.size/(2**i)+pos1[1])]=0.5
    #af(i==1):
     #   spec[spec.size/2]=1
      #  spec2[spec.size/(2)+pos1[2]]=0.5
       # spec1[spec.size/(2)-pos1[1]]=0.5
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
    spec_comb=spec+spec1+spec2
    """
    Portion to compute the echelle
    Contains a mask ech_plot.
    Later ech_plot is mutiplied with spec_comb to generate the actual ech_plot
    """
    freq_modulo=np.mod(Freq_axis,delt_nu)
    """
    ech_plot=np.reshape(np.zeros(Freq_axis.size*freq_bins.size),(Freq_axis.size,freq_bins.size))
    for j in range(N-1):
        for k in range(1,freq_bins.size):
            if((freq_modulo[j]<=freq_bins[k])&(freq_modulo[j]>=freq_bins[k-1])&(spec_comb[j]>0)):
            	ech_plot[j,k]=1.0
    """
    dat_frame=pd.DataFrame({'Frequency':Freq_axis,
			    'l0':spec,
			    'l1':spec1,
                            'l2':spec2,
                            'noise':rand_noise,
                            'GaussProf':gp})
    dat_frame.to_csv('/home/rakesh/Data/Spec_numax_%d_modes_%d.csv'%(numax, modes))
  

    """
    This is the plotting piece needs to be commented out later

    fig1=plt.plot(spec*gp,label='l=0 mode')
    plt.xticks(range(0,N,N/9),Freq_axis[range(0,N,N/9)])
    plt.xlabel('Frequencies($\mu$Hz)')
    plt.legend()
    plt.ylabel('Fake Power')
    plt.title('Only modes no noise')
    raw_input("Press enter to continue")
    plt.plot(spec1*gp,label='l=1 mode')
    plt.legend()
    raw_input("Press enter to continue");
    plt.plot(spec2*gp,label='l=2 mode')
    plt.legend()
    raw_input("Press enter to continue");
    plt.plot((spec+spec1+spec2)*gp+rand_noise)
    plt.title('Noise added')
    plt.ylabel('Noise power')
    raw_input("Press enter to continue");
    plt.plot(3*(spec+spec1+spec2)*gp,'r--')
    plt.plot(gp,'g--',label='Gaussian Profile')
    plt.title('Noise added with trace of actual modes')
    plt.legend()

    t0=np.zeros(ech_plot.shape)
    t1=np.zeros(ech_plot.shape)
    t2=np.zeros(ech_plot.shape)
    for i in range(ech_plot.shape[1]):
        t0[:,i]=ech_plot[:,i]*(spec)
        t1[:,i]=ech_plot[:,i]*(spec1)
    This is a tricky piece of code, I am finding points where 
    the echelle plot will be centered.
    This now will be stored for each mode that can 
    then be scatter plotted
    [loc_freq_0,loc_bins_0]=np.where(np.max(t0)==t0)
    [loc_freq_1,loc_bins_1]=np.where(np.max(t1)==t1)
    [loc_freq_2,loc_bins_2]=np.where(np.max(t2)==t2)

    plt.figure(2)
    plt.scatter(freq_bins[loc_bins_0],Freq_axis[loc_freq_0],c=t0[loc_freq_0,loc_bins_0],marker='^')
    plt.scatter(freq_bins[loc_bins_1],Freq_axis[loc_freq_1],c=t1[loc_freq_1,loc_bins_1],marker='*')
    plt.scatter(freq_bins[loc_bins_2],Freq_axis[loc_freq_2],c=t2[loc_freq_2,loc_bins_2],marker='>')
    plt.title('Echelle plot')
    plt.legend(['l=0','l=1','l=2'],loc='center')
    bins=[loc_bins_0,loc_bins_1,loc_bins_2]
    freqs=[loc_freq_0,loc_freq_1,loc_freq_2]
    plt.xlabel('$\\nu$ mod $\Delta\\nu$ $\mu$Hz')
    plt.ylabel('$\\nu$ $\mu$Hz')

    """










