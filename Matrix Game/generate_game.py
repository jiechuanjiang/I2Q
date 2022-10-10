import os, sys, time
import numpy as np
from buffer import ReplayBuffer
from config import *
from matrix import Matrix

def f(r_f, x, l, n, layer):

	if l == (layer-1):
		return r_f[x]
	else:
		max_t = -100000
		for i in range(n):
			max_t = max(f(r_f, x*n+i+1, l+1, n, layer),max_t)
		return r_f[x] + max_t

m = int(((n_actions*n_actions)**(layer+1) - 1)/((n_actions*n_actions)-1))
r_f = []
r = []
for i in range(100):

	r_f.append(2*np.random.rand(m) - 1)
	r.append(f(r_f[i], 0, 0, n_actions**2, layer))

r_f=np.array(r_f)
r=np.array(r)
np.save('r_f',r_f) 
np.save('best_return',r) 


