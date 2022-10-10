import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop

class QSS(th.nn.Module):
    def __init__(self, obs_len):
        super(QSS, self).__init__()
        self.fc1 = th.nn.Linear(2*obs_len, 256)
        self.fc2 = th.nn.Linear(256, 256)
        self.fc3 = th.nn.Linear(256, 1)

    def forward(self, x, y):

        q = self.fc3(F.relu(self.fc2(F.relu(self.fc1(th.cat((x,y),-1))))))
        return q

class VS(th.nn.Module):
    def __init__(self, obs_len):
        super(VS, self).__init__()
        self.fc1 = th.nn.Linear(obs_len, 256)
        self.fc2 = th.nn.Linear(256, 256)
        self.fc3 = th.nn.Linear(256, 1)

    def forward(self, x):

        q = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return q

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.qss = th.nn.ModuleList([QSS(self.args.rnn_hidden_dim) for _ in range(self.args.n_agents)]).cuda()
        self.vs = th.nn.ModuleList([VS(self.args.rnn_hidden_dim) for _ in range(self.args.n_agents)]).cuda()
        self.mixer = None
        
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_qss = RMSprop(params=self.qss.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_vs = RMSprop(params=self.vs.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        
        self.target_mac = copy.deepcopy(mac)
        self.target_qss = copy.deepcopy(self.qss)
        self.target_vs = copy.deepcopy(self.vs)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  

        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions == 0] = -9999999
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)        
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        targets = targets.detach()

        mask = mask.expand_as(targets).detach()
        mask_sum = mask.sum().detach()

        mac_hid = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_hid.append(self.target_mac.forward_hidden(batch, t=t))
        mac_hid = th.stack(mac_hid, dim=1).clone().detach()
        hid = mac_hid[:, :-1]
        next_hid = mac_hid[:,1:]

        v_s = th.stack([self.vs[i](hid[:,:,i]) for i in range(self.args.n_agents)], dim=2).squeeze(-1)
        target_v = th.stack([self.target_vs[i](next_hid[:,:,i])*self.args.gamma * (1 - terminated) + rewards for i in range(self.args.n_agents)], dim=2).squeeze(-1).detach()
        q_ss = th.stack([self.qss[i](hid[:,:,i],next_hid[:,:,i]) for i in range(self.args.n_agents)], dim=2).squeeze(-1)

        loss_qss = (((q_ss - target_v)**2)*mask).sum()/mask_sum
        self.optimiser_qss.zero_grad()
        loss_qss.backward()
        th.nn.utils.clip_grad_norm_(self.qss.parameters(), self.args.grad_norm_clip)
        self.optimiser_qss.step()

        q_ss = th.stack([self.target_qss[i](hid[:,:,i],next_hid[:,:,i]) for i in range(self.args.n_agents)], dim=2).squeeze(-1).detach()
        weights = ((q_ss > v_s).float()*(2*self.args.tau - 1) + 1 - self.args.tau).clone().detach()
        w_mask = weights*mask
        loss_vs = (((v_s - q_ss)**2)*w_mask).sum()/w_mask.sum()
        self.optimiser_vs.zero_grad()
        loss_vs.backward()
        th.nn.utils.clip_grad_norm_(self.vs.parameters(), self.args.grad_norm_clip)
        self.optimiser_vs.step()

        predicted_targets = (self.args.lamb*target_v+(1-self.args.lamb)*targets).detach()
        td_error = (chosen_action_qvals - predicted_targets)
        masked_td_error = (td_error**2) * mask
        loss = masked_td_error.sum() / mask.sum()
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (td_error.abs() * mask).sum().item()/(mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems), t_env)
            self.logger.log_stat("target_mean", (q_ss * mask).sum().item()/(mask_elems), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_qss.load_state_dict(self.qss.state_dict())
        self.target_vs.load_state_dict(self.vs.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
