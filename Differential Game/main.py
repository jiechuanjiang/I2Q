import os, sys, time
import numpy as np
import tensorflow as tf
from model import Agent
from buffer import ReplayBuffer
from config import *
from world import World
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

beta = float(sys.argv[1])
env = World(n_ant)
test_env = [World(n_ant) for _ in range(20)]

agents = Agent(sess,state_space,n_actions,n_ant,beta)
buff = ReplayBuffer(capacity,state_space,n_actions,n_ant)
agents.init_update()

def test():

	score, num, n_envs = 0, 0, 20
	r = np.zeros(n_envs)
	obs = np.ones((n_envs,state_space))
	for _ in range(10):
		for i in range(n_envs):
			o = test_env[i].reset()
			obs[i] = o

		for steps in range(100):
			action = np.zeros((n_envs,n_ant*n_actions))
			for i in range(n_ant):
				a_t = agents.actor[i].predict(np.array(obs),batch_size = 32)
				for k in range(n_envs):
					action[k][i*n_actions:(i+1)*n_actions] = a_t[k]

			for k in range(n_envs):
				next_obs, reward, terminated = test_env[k].step(action[k])
				obs[k] = next_obs
				r[k] += reward
	return sum(r)/200

f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt', 'w')
obs = env.reset()
epk = 0
while setps<max_steps:

	p = agents.acting.predict(np.array([obs]))
	for i in range(n_ant):
		if epk%4==0:
			p[i] = 2*np.random.rand(n_actions) - 1
		else:
			p[i] = np.clip(p[i][0] + 0.1*np.random.randn(n_actions),-1,1)
	next_obs, reward, terminated = env.step(np.hstack(p))
	setps += 1
	ep_len += 1
	if ep_len == max_ep_len:
		terminated = False
	buff.add(obs, np.array(p), reward, next_obs, terminated)
	obs = next_obs

	if (terminated)|(ep_len == max_ep_len):
		obs = env.reset()
		terminated = False
		ep_len = 0
		epk += 1

	if setps%10000==0:
		log_r = test()
		f.write(str(log_r)+'\n')
		f.flush()

	if (setps < 50000)|(setps%50!=0):
		continue

	for e in range(10):
		X, A, R, next_X, D = buff.getBatch(batch_size)
		agents.train(X, A, R, next_X, D)
		agents.update()