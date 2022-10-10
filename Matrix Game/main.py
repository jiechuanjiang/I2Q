import os, sys, time
import numpy as np
from buffer import ReplayBuffer
from config import *
from matrix import Matrix

index = int(sys.argv[1])
flag = float(sys.argv[2])
if flag == 1:
	ideal = True
m = int(((n_actions*n_actions)**(layer+1) - 1)/((n_actions*n_actions)-1))
best_return = np.load('best_return.npy')[index]
r_f = np.load('r_f.npy')[index]

f = open(sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'.txt', 'w')

env = Matrix(n_actions,layer,r_f)
test_env = Matrix(n_actions,layer,r_f)
buff = ReplayBuffer(capacity,state_space,n_actions,n_ant)

Q = []
Q_s = []
P = []
for i in range(n_ant):
	Q.append(np.random.rand(m,n_actions) - 0.5)
	Q_s.append(np.random.rand(m,n_actions*n_actions) - 0.5)
	P.append(-100000*np.ones((m,n_actions,n_actions*n_actions)))

def test_agent():
	o, d = test_env.reset(), False
	sum_reward = 0
	while not d:
		a = []
		for i in range(n_ant):
			a.append(np.argmax(Q[i][o]))
		o, r, d = test_env.step(a)
		sum_reward += r

	return sum_reward

obs = env.reset()
while setps<max_steps:

	a = []
	for i in range(n_ant):
		if np.random.rand() < epsilon:
			a.append(np.random.randint(n_actions))
		else:
			a.append(np.argmax(Q[i][obs]))
	next_obs, reward, terminated = env.step(a)
	setps += 1
	buff.add(obs, np.array(a).reshape(-1,1), reward, next_obs, terminated)
	for i in range(n_ant):
		P[i][obs][a[i]][next_obs - obs*n_actions*n_actions - 1] = 0
	obs = next_obs

	if terminated:
		obs = env.reset()
		terminated = False

	if setps%100==0:
		log_r = test_agent()
		f.write(str(log_r/best_return)+'\n')
		f.flush()

	if (setps < 10000)|(setps%50!=0):
		continue

	for e in range(10):
		X, A, R, next_X, D = buff.getBatch(batch_size)
		print(next_X.shape)
		exit()
		N_X = (next_X - X*n_actions*n_actions - 1).astype(np.int32)
		for i in range(n_ant):

			target = R + gamma*(1 - D)*Q_s[i][next_X].max(axis = 2)
			Q_s[i][X,N_X] = (1-alpha)*Q_s[i][X,N_X] + alpha*target
			if ideal:
				next_X = X*n_actions*n_actions + (Q_s[i][X]+P[i][X,A[i]]).argmax(axis = 2) + 1
				target = R + gamma*(1 - D)*Q[i][next_X].max(axis = 2)
			else:
				target = R + gamma*(1 - D)*Q[i][next_X].max(axis = 2)
			Q[i][X,A[i]] = (1-alpha)*Q[i][X,A[i]] + alpha*target











