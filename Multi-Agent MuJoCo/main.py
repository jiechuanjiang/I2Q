import os, sys, time
import numpy as np
import gym
import tensorflow as tf
from model import Agent
from buffer import ReplayBuffer
from config import *
from src.multiagent_mujoco.mujoco_multi import MujocoMulti
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

beta = float(sys.argv[1])
env_args = {"scenario": "Ant-v2",
              "agent_conf": "2x4",
              "agent_obsk": 0,
              "episode_limit": max_ep_len}
env = MujocoMulti(env_args=env_args)
test_env = MujocoMulti(env_args=env_args)
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_ant = env_info["n_agents"]
state_space = env_info["obs_shape"]


agents = Agent(sess,state_space,n_actions,n_ant,beta)
buff = ReplayBuffer(capacity,state_space,n_actions,n_ant)
agents.init_update()

def test_agent():
	sum_reward = 0
	for m in range(10):
		o, d, ep_l = test_env.reset(), False, 0
		while not(d or (ep_l == max_ep_len)):
			p = agents.acting.predict([np.array([o[i]]) for i in range(n_ant)])
			for i in range(n_ant):
				p[i] = p[i][0]
			r, d, _ = test_env.step(p)
			o = test_env.get_obs()
			sum_reward += r
			ep_l += 1
	return sum_reward/10

f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt', 'w')
obs = env.reset()
while setps<max_steps:
	p = agents.acting.predict([np.array([obs[i]]) for i in range(n_ant)])
	for i in range(n_ant):
		if setps < 10000:
			p[i] = 2*np.random.rand(n_actions) - 1
		else:
			p[i] = np.clip(p[i][0] + 0.1*np.random.randn(n_actions),-1,1)
	reward, terminated, info = env.step(p)
	next_obs = env.get_obs()
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

