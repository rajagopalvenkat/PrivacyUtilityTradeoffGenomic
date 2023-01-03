import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np
#from autograd import elementwise_grad, grad, jacobian
import math
import time
from threading import Thread
from tqdm import tqdm

theta=float(sys.argv[1])

bits=pickle.load(open('In_Pop.pkl', 'rb'))

d=[]

for b in bits:
	d+=[b.tolist()]
d=np.array(d, dtype=int)	

bits=[]

bits=pickle.load(open('Not_In_Pop.pkl', 'rb'))

d_n=[]

for b in bits:
	d_n+=[b.tolist()]
d_n=np.array(d_n, dtype=int)

bits=[]

#conbine d and d_n
d=np.concatenate((d, d_n), axis=0)
d_n=[]

#split d into train and test by sampling uniformly
np.random.seed(0)
inds = np.random.choice(np.arange(d.shape[0]), size=int(sys.argv[2]), replace=False)
d_train = d[inds]
d_test = np.delete(d, inds, axis=0)
d=[]

#rename d_train as d
d=d_train
d_train=[]
d_n=d_test
d_test=[]

n=int(sys.argv[2])
freqs_d = np.clip(np.mean(d, axis=0), a_min=0.0001, a_max=0.9999)
freqs_n = np.clip(np.mean(d_n, axis=0), a_min=0.0001, a_max=0.9999)

solutions={}

def get_best_DP(eps):
	global freqs_d
	global freqs_n
	global d
	global alpha
	global solutions

	masked=np.zeros(1338843, dtype=int)

	m=np.sum(masked==0)*(148515/1338843)
	delta = np.random.laplace(loc=0, scale=((m/(n*eps))), size=1338843)
	freqs_noisy=np.clip(freqs_d+delta, a_min=0.0001, a_max=0.9999)

	Delta = ((d*(np.log(freqs_noisy/freqs_n))) + ((1-d)*(np.log((1-freqs_noisy)/(1-freqs_n)))))

	sorted_snvs = np.argsort(np.mean(Delta, axis=0))

	i=0

	best_obj = np.inf
	sol_dict={'cov':0, 'norm':0, 'mask':0}

	print('iterating for eps ', eps)

	while i<=1338000:
		cov=np.sum( np.sum((Delta[:,masked==0]), axis=1) <= theta )
		norm_noise = np.linalg.norm(delta[masked==0], ord=1)
		
		if eps in solutions:
			solutions[eps] += [(cov, i, norm_noise),]
		else:
			solutions[eps] = [ (cov, i, norm_noise) ]

		masked[sorted_snvs[i:i+1000]]=1
		i+=1000


threads=list()

for eps in [10000000, 5000000, 1000000, 500000, 100000, 10000, 1000]:
	threads.append(Thread(target=get_best_DP, args=(eps,)))
for thread in threads:
	thread.start()
for thread in threads:
	thread.join()

for k,v in solutions.items():
	print(k, v)

