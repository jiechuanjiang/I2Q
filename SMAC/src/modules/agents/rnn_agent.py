import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.ModuleList([nn.Linear(input_shape, args.rnn_hidden_dim) for _ in range(self.args.n_agents)])
        self.rnn = nn.ModuleList([nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim) for _ in range(self.args.n_agents)])
        self.fc2 = nn.ModuleList([nn.Linear(args.rnn_hidden_dim, args.n_actions) for _ in range(self.args.n_agents)])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        inputs = inputs.reshape(-1, self.args.n_agents, inputs.shape[1])
        x = [F.relu(self.fc1[i](inputs[:,i])) for i in range(self.args.n_agents)]
        h_in = hidden_state.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        h = [self.rnn[i](x[i], h_in[:,i]) for i in range(self.args.n_agents)]
        q = [self.fc2[i](h[i]) for i in range(self.args.n_agents)]
        h = torch.cat(h,1).reshape(-1,self.args.rnn_hidden_dim)
        q = torch.cat(q,1).reshape(-1,self.args.n_actions)
        return q, h
