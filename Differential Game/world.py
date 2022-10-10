import numpy as np
import copy

def f(x, m):

	if x < m:
		return 0.5*(np.cos(3.14/m*x)+1)
	elif x < 0.6:
		return 0
	elif x < 1.0:
		return 0.15*(np.cos(15*(x-0.8))+1)
	else:
		return 0

class World(object):
	def __init__(self, n):
		super(World, self).__init__()
		self.n_agent = n
		self.n_action = 1
		self.len_state = n
		self.x = np.random.rand(self.n_agent)*2-1
		self.m_list = [0.0,0.1,0.12,0.25,0.38,0.51]
		self.m = self.m_list[n]

	def reset(self):

		self.x = np.random.rand(self.n_agent)*2-1

		return self.x

	def get_state(self):

		return self.x


	def step(self,actions):

		done = False
		reward = f(np.sqrt(2/self.n_agent*sum(self.x**2)),self.m)
		self.x = np.clip(self.x + 0.1*actions,-1,1)

		return self.x, reward, done
