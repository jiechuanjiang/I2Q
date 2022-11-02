import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Input, Dense, Concatenate, Add, Reshape
from keras.models import Model

def build_actor(num_features,n_actions,a_limit):

	I1 = Input(shape = (num_features,))
	h1 = Dense(256,activation = 'relu')(I1) 
	h2 = Dense(256,activation = 'relu')(h1)
	V = Dense(n_actions,activation = 'tanh')(h2)
	model = Model(I1, V)
	V = Lambda(lambda x : x*a_limit)(V)

	return model

def build_critic(state_space,n_actions):

	Inputs = []
	Inputs.append(Input(shape = (state_space,)))
	Inputs.append(Input(shape = (n_actions,)))

	I = Concatenate(axis=1)(Inputs)
	h = Dense(256,activation = 'relu')(I)
	h = Dense(256,activation = 'relu')(h)
	q_total = Dense(1)(h)
	model = Model(Inputs, q_total)

	return model

def build_qss(state_space):

	Inputs = []
	Inputs.append(Input(shape = (state_space,)))
	Inputs.append(Input(shape = (state_space,)))

	I = Concatenate(axis=1)(Inputs)
	h = Dense(256,activation = 'relu')(I)
	h = Dense(256,activation = 'relu')(h)
	q_total = Dense(1)(h)
	model = Model(Inputs, q_total)

	return model

def build_vs(state_space):

	Inputs = Input(shape = (state_space,))

	h = Dense(256,activation = 'relu')(Inputs)
	h = Dense(256,activation = 'relu')(h)
	q_total = Dense(1)(h)
	model = Model(Inputs, q_total)

	return model

def build_Q_tot(state_space,actor,critic):

	Inputs = Input(shape=[state_space])
	q_value = critic([Inputs,actor(Inputs)])
	model = Model(Inputs, q_value)

	return model

def build_acting(state_space,actors,n_ant):

	Inputs = Input(shape=[state_space])
	actions = []
	for i in range(n_ant):
		actions.append(actors[i](Inputs))
	model = Model(Inputs, actions)

	return model


class Agent(object):
	def __init__(self,sess,state_space,n_actions,n_ant,a_limit,tau):
		super(Agent, self).__init__()
		self.sess = sess
		self.n_actions = n_actions
		self.n_ant = n_ant
		self.state_space = state_space
		self.gamma = 0.99
		self.tau = tau
		self.a_limit = a_limit
		self.update_type = 1 # 0: IQL 1: I2Q
		K.set_session(sess)
		
		self.actor = []
		self.critic = []
		self.Q_tot = []
		self.qss = []
		self.vs = []
		self.critic_i2q = []
		for i in range(self.n_ant):
			self.actor.append(build_actor(self.state_space,self.n_actions,self.a_limit))
			self.critic.append(build_critic(self.state_space,self.n_actions))
			self.Q_tot.append(build_Q_tot(self.state_space,self.actor[i],self.critic[i]))
			self.qss.append(build_qss(self.state_space))
			self.vs.append(build_vs(self.state_space))
			self.critic_i2q.append(build_critic(self.state_space,self.n_actions))
		self.acting = build_acting(self.state_space,self.actor,self.n_ant)

		self.actor_tar = []
		self.critic_tar = []
		self.Q_tot_tar = []
		self.qss_tar = []
		self.critic_i2q_tar = []
		for i in range(self.n_ant):
			self.actor_tar.append(build_actor(self.state_space,self.n_actions,self.a_limit))
			self.critic_tar.append(build_critic(self.state_space,self.n_actions))
			self.Q_tot_tar.append(build_Q_tot(self.state_space,self.actor_tar[i],self.critic_tar[i]))
			self.qss_tar.append(build_qss(self.state_space))
			self.critic_i2q_tar.append(build_critic(self.state_space,self.n_actions))

		self.S = []
		self.A = []
		self.Next_S = []
		self.R = tf.placeholder(tf.float32,[None, 1])
		self.D = tf.placeholder(tf.float32,[None, 1])
		for i in range(self.n_ant):
			self.S.append(tf.placeholder(tf.float32,[None, self.state_space]))
			self.A.append(tf.placeholder(tf.float32,[None, self.n_actions]))
			self.Next_S.append(tf.placeholder(tf.float32,[None, self.state_space]))

		self.opt_actor = []
		self.opt_critic = []
		self.opt_qss = []
		self.opt_vs = []
		self.opt_critic_i2q = []

		for i in range(self.n_ant):
			
			if self.update_type == 0: #IQL
				self.opt_actor.append(tf.train.AdamOptimizer(0.001).minimize(-tf.reduce_mean(self.Q_tot[i](self.S[i])),var_list = self.actor[i].trainable_weights))
				Q_target = tf.stop_gradient(self.R + self.gamma*(1 - self.D)*self.Q_tot_tar[i](self.Next_S[i]))
				self.opt_critic.append(tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.critic[i]([self.S[i],self.A[i]]) - Q_target)**2),var_list = self.critic[i].trainable_weights))
			
			else: #I2Q
				VS_target = tf.stop_gradient(self.R + self.gamma*(1 - self.D)*self.vs[i](self.Next_S[i]))
				QSS_target = tf.stop_gradient(self.qss_tar[i]([self.S[i],self.Next_S[i]]))
				self.opt_qss.append(tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.qss[i]([self.S[i],self.Next_S[i]]) - VS_target)**2),var_list = self.qss[i].trainable_weights))

				VS = self.vs[i](self.S[i])
				weights = tf.stop_gradient(tf.to_float(QSS_target > VS)*(2*self.tau - 1) + 1 - self.tau)
				self.opt_vs.append(tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean(weights*(VS - QSS_target)**2),var_list = self.vs[i].trainable_weights))
				
				Q_target = tf.stop_gradient(self.R + self.gamma*(1 - self.D)*self.Q_tot_tar[i](self.Next_S[i]))
				self.opt_critic.append(tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.critic[i]([self.S[i],self.A[i]]) - Q_target)**2),var_list = self.critic[i].trainable_weights))

				Q_i2q_target = self.R + self.gamma*(1 - self.D)*self.critic_i2q_tar[i]([self.Next_S[i],self.actor_tar[i](self.Next_S[i])])
				Q_i2q_target = tf.stop_gradient(tf.maximum(VS_target,Q_i2q_target))
				self.opt_critic_i2q.append(tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.critic_i2q[i]([self.S[i],self.A[i]]) - Q_i2q_target)**2),var_list = self.critic_i2q[i].trainable_weights))
				
				self.opt_actor.append(tf.train.AdamOptimizer(0.001).minimize(-0.5*tf.reduce_mean(self.Q_tot[i](self.S[i])) - 0.5*tf.reduce_mean(self.critic_i2q[i]([self.S[i],self.actor[i](self.S[i])])),var_list = self.actor[i].trainable_weights))

		self.opt_actor = tf.group(self.opt_actor)
		self.opt_critic = tf.group(self.opt_critic)
		self.opt_vs = tf.group(self.opt_vs)
		self.opt_qss = tf.group(self.opt_qss)
		self.opt_critic_i2q = tf.group(self.opt_critic_i2q)
		self.opt = tf.group([self.opt_qss,self.opt_vs,self.opt_critic_i2q,self.opt_critic,self.opt_actor])

		self.soft_replace = []
		for i in range(self.n_ant):
			self.soft_replace += [tf.assign(tar, 0.995*tar + (1 - 0.995)*main) for tar, main in zip(self.Q_tot_tar[i].trainable_weights, self.Q_tot[i].trainable_weights)]
			if self.update_type != 0:
				self.soft_replace += [tf.assign(tar, 0.995*tar + (1 - 0.995)*main) for tar, main in zip(self.qss_tar[i].trainable_weights, self.qss[i].trainable_weights)]
				self.soft_replace += [tf.assign(tar, 0.995*tar + (1 - 0.995)*main) for tar, main in zip(self.critic_i2q_tar[i].trainable_weights, self.critic_i2q[i].trainable_weights)]
		self.soft_replace = tf.group(self.soft_replace)

		self.sess.run(tf.global_variables_initializer())

	def train(self, S, A, R, Next_S, D):

		dict_t = {}
		for i in range(self.n_ant):
			dict_t[self.A[i]] = A[i]
			dict_t[self.S[i]] = S
			dict_t[self.Next_S[i]] = Next_S
		dict_t[self.R] = R
		dict_t[self.D] = D
		return self.sess.run(self.opt, feed_dict=dict_t)

	def update(self):
		self.sess.run(self.soft_replace)

	def init_update(self):
		for i in range(self.n_ant):
			self.Q_tot_tar[i].set_weights(self.Q_tot[i].get_weights())
			self.qss_tar[i].set_weights(self.qss[i].get_weights())
			self.critic_i2q_tar[i].set_weights(self.critic_i2q[i].get_weights())

		

