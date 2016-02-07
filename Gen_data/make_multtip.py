def make_multtrip():
	import numpy as np
	import pandas as pd
	from scipy.stats import norm
	from scipy.stats import cauchy
	freqs=np.arange(1800,4000,4)
	N=freqs.size
	lp=cauchy(0).pdf(np.arange(-1,1,1./2))
	nu_max=0
	offset=30
	other_mode=25
	#act_offset=np.random.randint(10,high=20)
	for i in range(offset+lp.size+5,N-lp.size-5-offset):
		#act_offset=np.random.randint(0,high=20)
		act_offset=8#fixed distance
		spec=np.zeros(N)
		spec[i-lp.size/2:i+lp.size/2]=2*lp
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos_off=pos-act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.


		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos+=other_mode
		spec[pos]=2*lp
		pos_off=pos-act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
		
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos-=other_mode
		spec[pos]=2*lp
		spec[i-lp.size-5:i-5]=1*lp
	#	pos=0.
	#	pos=np.arange(i-lp.size-5,i-5,1)
		pos_off=pos-act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
	#	spec[pos]=0
		#act_offset=np.random.randint(0,high=20)
		#act_offset=5#fixed distance
		#pos=0.
		#spec[i+5:i+lp.size+5]=1*lp
	#	pos=np.arange(i+5,i+lp.size+5,1)
	#	pos_off=pos+act_offset
	#	spec[pos_off]=spec[pos]
	#	spec[pos]=0
		rand_noise=np.random.random(N)*np.random.randint(1,high=2)
		
		df=pd.DataFrame({'Frequency':freqs,
				 'l0':spec,
				 'noise':rand_noise,})
		df.to_csv('/home/rakesh/Fake_Data/MultiTrip/Spec_numax_%d.csv'%(freqs[i]))
def make_sing_multtrip(nu_max):
	import numpy as np
	import pandas as pd
	from scipy.stats import norm
	from scipy.stats import cauchy
	freqs=np.arange(1800,4000,4)
	N=freqs.size
	lp=cauchy(0).pdf(np.arange(-1,1,1./2))
	offset=30
	other_mode=25
	arr=np.where(nu_max==freqs)
	i=arr[0]
	#act_offset=np.random.randint(10,high=20)
	#for i in range(offset+lp.size+5,N-lp.size-5-offset):
		#act_offset=np.random.randint(0,high=20)
	act_offset=8#fixed distance
	spec=np.zeros(N)
	spec[i-lp.size/2:i+lp.size/2]=2*lp
	pos=np.arange(i-lp.size/2,i+lp.size/2)
	pos_off=pos-act_offset
	spec[pos_off]=0.5*spec[pos]
	pos_off=0.
	pos_off=pos+act_offset
	spec[pos_off]=0.5*spec[pos]
	pos_off=0.


	pos=np.arange(i-lp.size/2,i+lp.size/2)
	pos+=other_mode
	spec[pos]=2*lp
	pos_off=pos-act_offset
	spec[pos_off]=0.5*spec[pos]
	pos_off=0.
	pos_off=pos+act_offset
	spec[pos_off]=0.5*spec[pos]
	pos_off=0.
	
	pos=np.arange(i-lp.size/2,i+lp.size/2)
	pos-=other_mode
	spec[pos]=2*lp
	spec[i-lp.size-5:i-5]=1*lp
#	pos=0.
#	pos=np.arange(i-lp.size-5,i-5,1)
	pos_off=pos-act_offset
	spec[pos_off]=0.5*spec[pos]
	pos_off=0.
	pos_off=pos+act_offset
	spec[pos_off]=0.5*spec[pos]
	pos_off=0.
#	spec[pos]=0
	#act_offset=np.random.randint(0,high=20)
	#act_offset=5#fixed distance
	rand_noise=np.random.random(N)*np.random.randint(1,high=2)
	sig=spec
	sig/=np.max(sig)
	sig_noise=spec+rand_noise
	sig_noise/=np.max(sig_noise)
	return(sig,sig_noise)
def make_twins():
	import numpy as np
	import pandas as pd
	from scipy.stats import norm
	from scipy.stats import cauchy
	freqs=np.arange(1800,4000,4)
	N=freqs.size
	lp=cauchy(0).pdf(np.arange(-1,1,1./2))
	nu_max=0
	offset=30
	other_mode=50
	peaks=abs(offset+lp.size+5-(N-lp.size-5-offset))
        #act_offset=np.random.randint(10,high=20)
        path="/home/rakesh/documents/TIFR/Data/Train_twins/"
        sig=np.zeros(peaks*(N+other_mode)).reshape(peaks,(N+other_mode))
        sig_noise=np.zeros(peaks*(N+other_mode)).reshape(peaks,(N+other_mode))
	for i in range(offset+lp.size+5,N-lp.size-5-offset):
		#act_offset=np.random.randint(0,high=20)
		act_offset=8#fixed distance
		spec=np.zeros(N+other_mode)
		spec[i-lp.size/2:i+lp.size/2]=2*lp
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos_off=pos-act_offset+3
		spec[pos_off]=0.2*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset+3
		spec[pos_off]=0.8*spec[pos]
		pos_off=0.


		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos+=other_mode
		spec[pos]=2*lp
		pos_off=pos-act_offset+3
		spec[pos_off]=0.2*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset+3
		spec[pos_off]=0.8*spec[pos]
		pos_off=0.
                """	
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos-=other_mode
		spec[pos]=2*lp
		spec[i-lp.size-5:i-5]=1*lp
	#	pos=0.
	#	pos=np.arange(i-lp.size-5,i-5,1)
		pos_off=pos-act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
	#	spec[pos]=0
		#act_offset=np.random.randint(0,high=20)
		#act_offset=5#fixed distance
		#pos=0.
		#spec[i+5:i+lp.size+5]=1*lp
	#	pos=np.arange(i+5,i+lp.size+5,1)
	#	pos_off=pos+act_offset
	#	spec[pos_off]=spec[pos]
                """
		rand_noise=np.random.random(N+other_mode)*np.random.randint(1,high=2)
                sig[i-(offset+lp.size+5),:]=spec
                sig[i-(offset+lp.size+5),:]/=np.max(sig[i-(offset+lp.size+5)])
                sig_noise[i-(offset+lp.size+5),:]=(spec+rand_noise)
                sig_noise[i-(offset+lp.size+5),:]/=np.max(sig_noise[i-(offset+lp.size+5)])

        np.save(path+"sig.npy",sig)
        np.save(path+"sig_noise.npy",sig_noise)


def make_twins_test():
	import numpy as np
	import pandas as pd
	from scipy.stats import norm
	from scipy.stats import cauchy
        import os
	freqs=np.arange(1800,4000,4)
	N=freqs.size
	lp=cauchy(0).pdf(np.arange(-1,1,1./2))
	nu_max=0
	offset=30
	other_mode=50
	peaks=abs(offset+lp.size+5-(N-lp.size-5-offset))
        #act_offset=np.random.randint(10,high=20)
        path="/home/rakesh/documents/TIFR/Data/Test_twins/"
        if(os.path.isdir(path)==False):
            os.makedirs(path)

        sig=np.zeros(peaks*(N+other_mode)).reshape(peaks,(N+other_mode))
        sig_noise=np.zeros(peaks*(N+other_mode)).reshape(peaks,(N+other_mode))
	for i in range(offset+lp.size+5,N-lp.size-5-offset):
		#act_offset=np.random.randint(0,high=20)
		act_offset=8#fixed distance
		spec=np.zeros(N+other_mode)
		spec[i-lp.size/2:i+lp.size/2]=2*lp
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos_off=pos-act_offset+3
		spec[pos_off]=0.2*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset+3
		spec[pos_off]=0.8*spec[pos]
		pos_off=0.


		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos+=other_mode+np.random.randint(1,high=10)
		spec[pos]=2*lp
		pos_off=pos-act_offset+3
		spec[pos_off]=0.2*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset-3
		spec[pos_off]=0.8*spec[pos]
		pos_off=0.
                """	
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos-=other_mode
		spec[pos]=2*lp
		spec[i-lp.size-5:i-5]=1*lp
	#	pos=0.
	#	pos=np.arange(i-lp.size-5,i-5,1)
		pos_off=pos-act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
	#	spec[pos]=0
		#act_offset=np.random.randint(0,high=20)
		#act_offset=5#fixed distance
		#pos=0.
		#spec[i+5:i+lp.size+5]=1*lp
	#	pos=np.arange(i+5,i+lp.size+5,1)
	#	pos_off=pos+act_offset
	#	spec[pos_off]=spec[pos]
                """
		rand_noise=np.random.random(N+other_mode)*np.random.randint(1,high=2)
                sig[i-(offset+lp.size+5),:]=spec
                sig[i-(offset+lp.size+5),:]/=np.max(sig[i-(offset+lp.size+5)])
                sig_noise[i-(offset+lp.size+5),:]=(spec+rand_noise)
                sig_noise[i-(offset+lp.size+5),:]/=np.max(sig_noise[i-(offset+lp.size+5)])

        np.save(path+"sig.npy",sig)
        np.save(path+"sig_noise.npy",sig_noise)

def make_twins_cv():
	import numpy as np
	import pandas as pd
	from scipy.stats import norm
	from scipy.stats import cauchy
        import os
	freqs=np.arange(1800,4000,4)
	N=freqs.size
	lp=cauchy(0).pdf(np.arange(-1,1,1./2))
	nu_max=0
	offset=30
	other_mode=50
	peaks=abs(offset+lp.size+5-(N-lp.size-5-offset))
        #act_offset=np.random.randint(10,high=20)
        path="/home/rakesh/documents/TIFR/Data/CV_twins/"
        if(os.path.isdir(path)==False):
            os.makedirs(path)

        sig=np.zeros(peaks*(N+other_mode)).reshape(peaks,(N+other_mode))
        sig_noise=np.zeros(peaks*(N+other_mode)).reshape(peaks,(N+other_mode))
	for i in range(offset+lp.size+5,N-lp.size-5-offset):
		#act_offset=np.random.randint(0,high=20)
		act_offset=8#fixed distance
		spec=np.zeros(N+other_mode)
		spec[i-lp.size/2:i+lp.size/2]=2*lp
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos_off=pos-act_offset+3
		spec[pos_off]=0.2*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset+3
		spec[pos_off]=0.8*spec[pos]
		pos_off=0.


		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos+=other_mode+np.random.randint(1,high=10)
		spec[pos]=2*lp
		pos_off=pos-act_offset+3
		spec[pos_off]=0.2*spec[pos]*np.random.randint(0,high=1)
		pos_off=0.
		pos_off=pos+act_offset+3
		spec[pos_off]=0.8*spec[pos]		
                pos_off=0.
                """	
		pos=np.arange(i-lp.size/2,i+lp.size/2)
		pos-=other_mode
		spec[pos]=2*lp
		spec[i-lp.size-5:i-5]=1*lp
	#	pos=0.
	#	pos=np.arange(i-lp.size-5,i-5,1)
		pos_off=pos-act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
		pos_off=pos+act_offset
		spec[pos_off]=0.5*spec[pos]
		pos_off=0.
	#	spec[pos]=0
		#act_offset=np.random.randint(0,high=20)
		#act_offset=5#fixed distance
		#pos=0.
		#spec[i+5:i+lp.size+5]=1*lp
	#	pos=np.arange(i+5,i+lp.size+5,1)
	#	pos_off=pos+act_offset
	#	spec[pos_off]=spec[pos]
                """
		rand_noise=np.random.random(N+other_mode)*np.random.randint(1,high=2)
                sig[i-(offset+lp.size+5),:]=spec
                sig[i-(offset+lp.size+5),:]/=np.max(sig[i-(offset+lp.size+5)])
                sig_noise[i-(offset+lp.size+5),:]=(spec+rand_noise)
                sig_noise[i-(offset+lp.size+5),:]/=np.max(sig_noise[i-(offset+lp.size+5)])

        np.save(path+"sig.npy",sig)
        np.save(path+"sig_noise.npy",sig_noise)

