import os, sys, time
import numpy as np
import gym
import tensorflow as tf
from model import Agent
from buffer import ReplayBuffer
from config import *
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

tau = float(sys.argv[1])
env = gym.make('Walker2d-v3')
test_env = gym.make('Walker2d-v3')

agents = Agent(sess,state_space,n_actions,n_ant,a_limit,tau)
buff = ReplayBuffer(capacity,state_space,n_actions,n_ant)
agents.init_update()

def test_agent():
	sum_reward = 0
	for m in range(10):
		o, d, ep_l = test_env.reset(), False, 0
		o = o[0:state_space]
		while not(d or (ep_l == max_ep_len)):
			p = agents.acting.predict(np.array([o]))
			for i in range(n_ant):
				p[i] = p[i][0]
			o, r, d, _ = test_env.step(np.hstack(p))
			o = o[0:state_space]
			sum_reward += r
			ep_l += 1
	return sum_reward/10

f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt', 'w')
obs = env.reset()
obs = obs[0:state_space]
while setps<max_steps:

	p = agents.acting.predict(np.array([obs]))
	for i in range(n_ant):
		if setps < 10000:
			p[i] = 2*np.random.rand(n_actions) - 1
		else:
			p[i] = np.clip(p[i][0] + 0.1*np.random.randn(n_actions),-1,1)
	next_obs, reward, terminated, info = env.step(np.hstack(p))
	next_obs = next_obs[0:state_space]
	setps += 1
	ep_len += 1
	if ep_len == max_ep_len:
		terminated = False
	buff.add(obs, np.array(p), reward, next_obs, terminated)
	obs = next_obs

	if (terminated)|(ep_len == max_ep_len):
		obs = env.reset()
		obs = obs[0:state_space]
		terminated = False
		ep_len = 0

	if setps%10000==0:
		log_r = test_agent()
		f.write(str(log_r)+'\n')
		f.flush()

	if (setps < 1000)|(setps%50!=0):
		continue

	for e in range(50):
		X, A, R, next_X, D = buff.getBatch(batch_size)
		agents.train(X, A, R, next_X, D)
		agents.update()

