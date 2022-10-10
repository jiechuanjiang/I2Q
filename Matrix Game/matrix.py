class Matrix(object):
	def __init__(self, n, l, r):
		super(Matrix, self).__init__()
		self.n = n 
		self.l = l 
		self.r = r
		self.s = 0
		self.t = 0

	def reset(self):
		self.s = 0
		self.t = 0
		return int(self.s)

	def get_state(self):

		return int(self.s)

	def step(self,actions):

		self.t += 1
		reward = self.r[int(self.s)]
		done = True if (self.t == self.l) else False

		i = actions[0]*self.n + actions[1] + 1
		self.s = self.s*(self.n*self.n) + i

		return int(self.s),reward,done

	
