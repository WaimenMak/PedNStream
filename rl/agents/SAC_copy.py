# -*- coding: utf-8 -*-
# @Time    : 05/01/2026 16:45
# @Author  : mmai
# @FileName: SAC.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch.nn as nn
from rl.rl_utils import ReplayBuffer, save_with_best_return
from tqdm import tqdm
import collections


class LSTMPolicyNetContinuous(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1, action_bound=2.5):
        super(LSTMPolicyNetContinuous, self).__init__()
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_size, act_dim)
        self.fc_std = nn.Linear(hidden_size, act_dim)
        self.action_bound = action_bound

    def forward(self, x, hidden=None):
        # x = F.relu(self.lstm(x))
        lstm_out, hidden_out = self.lstm(x, hidden)
        mu = self.fc_mu(lstm_out)
        std = F.softplus(self.fc_std(lstm_out))

        return mu, std, hidden_out

class LSTMQValueNetContinuous(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1):
        super(LSTMQValueNetContinuous, self).__init__()
        self.lstm = nn.LSTM(
            input_size=obs_dim + act_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, a, hidden=None):
        x = torch.cat([x, a], dim=1)
        lstm_out, hidden_out = self.lstm(x, hidden)
        x = F.relu(self.fc2(lstm_out))
        value = self.fc_out(x)
        return value, hidden_out

class StackedEncoder(torch.nn.Module):
    def __init__(self, feat_dim, stack_size, hidden_size=64, kernel_size=3):
        super(StackedEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feat_dim,
                               out_channels=hidden_size,
                               kernel_size=kernel_size,
                               stride=1, padding=0)
        # self.fc = nn.Linear((stack_size - kernel_size + 1) * hidden_size, hidden_size)
        self.ouput_dim = (stack_size - kernel_size + 1) * hidden_size
    def forward(self, x):

        x = F.elu(self.conv1(x))
        x = x.flatten(start_dim=1)
        return x

class MLPEncoder(torch.nn.Module):
    def __init__(self, feat_dim, stack_size=4, hidden_size=64):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(feat_dim * stack_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ouput_dim = hidden_size

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, stack_size=4, hidden_size=64, kernel_size=3, action_bound=2.5):
        super(PolicyNetContinuous, self).__init__()
        self.feat_dim = obs_dim // act_dim
        self.seq_len = stack_size
        self.num_links = act_dim
        # self.encoder = StackedEncoder(obs_dim, stack_size=stack_size, hidden_size=hidden_size, kernel_size=kernel_size)
        self.encoder = MLPEncoder(self.feat_dim, stack_size=stack_size, hidden_size=hidden_size)
        self.link_model = nn.Linear(self.encoder.ouput_dim, hidden_size)
        self.ud_model = nn.Linear(2*hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, 1) # one action per link
        self.fc_std = nn.Linear(hidden_size, 1) # one std per link
        self.action_bound = action_bound
        # self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (batch, seq, feat)
        if x.dim() == 2:
            x = x.unsqueeze(0) # batch size 1

        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.num_links, self.feat_dim).transpose(1, 2)
        zs = self.encoder(x.reshape(batch_size, self.num_links, -1)) # (batch_size, num_links, hidden_size)
        link_features = self.link_model(zs)
        all_links_sum = link_features.sum(dim=1)
        other_links_features = all_links_sum.unsqueeze(1) - link_features
        combined_features = torch.cat([link_features, other_links_features], dim=2)
        ud_features = self.ud_model(combined_features)
        # ud_features = ud_features.mean(dim=1)
        mu = self.fc_mu(F.relu(ud_features))
        std = F.softplus(self.fc_std(F.relu(ud_features)))
        return mu, std

class QValueNetContinuous(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, stack_size=4, hidden_size=64, kernel_size=3, action_bound=2.5):
        super(QValueNetContinuous, self).__init__()
        self.feat_dim = obs_dim // act_dim
        self.num_links = act_dim
        self.seq_len = stack_size
        # self.encoder = StackedEncoder(obs_dim, stack_size=stack_size, hidden_size=hidden_size, kernel_size=kernel_size)
        self.encoder = MLPEncoder(self.feat_dim, stack_size=stack_size, hidden_size=hidden_size)
        self.link_model = nn.Linear(self.encoder.ouput_dim, hidden_size)
        self.ud_model = nn.Linear(2*hidden_size, hidden_size)
        self.global_model = nn.Linear(hidden_size + act_dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        if s.dim() == 2:
            s = s.unsqueeze(0) # batch size 1

        batch_size = s.shape[0]
        s = s.view(batch_size, self.seq_len, self.num_links, self.feat_dim).transpose(1, 2)
        zs = self.encoder(s.reshape(batch_size, self.num_links, -1)) # (batch_size, num_links, hidden_size)
        link_features = self.link_model(zs)
        all_links_sum = link_features.sum(dim=1)
        other_links_features = all_links_sum.unsqueeze(1) - link_features
        combined_features = torch.cat([link_features, other_links_features], dim=2)
        ud_features = self.ud_model(combined_features)
        global_features = ud_features.mean(dim=1)
        # gate_widths = s[:, -1, -1].unsqueeze(1)  # get the last time step gate widths
        features = torch.cat([global_features, a.squeeze()], dim=1)
        features = self.global_model(features)
        value = self.fc_out(F.elu(features))
        return value

def train_off_policy_multi_agent(
    env,
    agents,
    num_episodes=100,
    delta_actions=False,
    minimal_size=500,
    batch_size=64,
    stack_size=4,
    randomize=False,
    seed=None,
    agents_saved_dir=None
):
    """
    Train SAC agents off-policy.
    """
    # Initialize return tracking for each agent
    return_dict = {agent_id: [] for agent_id in agents.keys()}
    global_episode = 0  # Track global episode count for saving

    # Track best average return across all agents (initialize to negative infinity)
    best_avg_return = float('-inf')
    # history queue for states stack
    first_agent_id = next(iter(agents))
    stack_size = agents[first_agent_id].stack_size
    state_history_queue = {agent_id: collections.deque(maxlen=stack_size) for agent_id in agents.keys()}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_returns = {agent_id: 0.0 for agent_id in agents.keys()}
                episode_true_returns = {agent_id: 0.0 for agent_id in agents.keys()}  # Track true (un-normalized) rewards
                obs, infos = env.reset(seed=seed, options={'randomize': randomize})
                # initialize state history queue
                state_stack = {}
                for agent_id in agents.keys():
                    state_history_queue[agent_id].clear()
                    for _ in range(stack_size):
                        state_history_queue[agent_id].append(obs[agent_id])
                    state_stack[agent_id] = np.array(state_history_queue[agent_id])


                done = False
                step = 0 # only for progress bar
                while not done:
                    actions = {}
                    absolute_actions = {}
                    for agent_id, agent in agents.items():
                        # action = agent.take_action(obs[agent_id])
                        action = agent.take_action(state_stack[agent_id])
                        if delta_actions:
                            absolute_action = obs[agent_id].reshape(agents[agent_id].act_dim, -1)[:,-1] + action
                            absolute_action = np.clip(absolute_action, agents[agent_id].act_low, agents[agent_id].act_high)
                            absolute_actions[agent_id] = absolute_action
                        else:
                            absolute_actions[agent_id] = action
                        actions[agent_id] = action
                    next_obs, rewards, terms, truncs, infos = env.step(absolute_actions)
                    next_state_stack = {}
                    for agent_id in agents.keys():
                        state_history_queue[agent_id].append(next_obs[agent_id])
                        next_state_stack[agent_id] = np.array(state_history_queue[agent_id])
                        agent.replay_buffer.add(state_stack[agent_id], actions[agent_id], rewards[agent_id], next_state_stack[agent_id], terms[agent_id])
                        episode_returns[agent_id] += rewards[agent_id]
                        # Track true (un-normalized) rewards if available
                        if agent_id in infos and 'true_reward' in infos[agent_id]:
                            episode_true_returns[agent_id] += infos[agent_id]['true_reward']
                        else:
                            episode_true_returns[agent_id] += rewards[agent_id]
                        # update agent
                        if agent.replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = agent.replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                            agent.update(transition_dict)
                    # for agent_id in agents.keys():
                    #     agent.replay_buffer.add(obs[agent_id], actions[agent_id], rewards[agent_id], next_obs[agent_id], terms[agent_id])
                    #     episode_returns[agent_id] += rewards[agent_id]

                    obs = next_obs
                    state_stack = next_state_stack

                    step += 1

                    done = any(terms.values()) or any(truncs.values())

                for agent_id in agents.keys():
                    return_dict[agent_id].append(episode_returns[agent_id])


                global_episode += 1
                if agents_saved_dir and  global_episode > num_episodes/2:
                    best_avg_return = save_with_best_return(agents, agents_saved_dir, episode_returns=episode_true_returns, best_avg_return=best_avg_return, global_episode=global_episode)
                # Update progress bar with both normalized and true returns
                if (i_episode+1) % 10 == 0:
                    avg_return = np.mean([np.mean(return_dict[aid][-10:]) for aid in agents.keys()])
                    avg_true_return = np.mean(list(episode_true_returns.values()))
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'norm_ret': '%.3f' % avg_return,
                        'true_ret': '%.3f' % avg_true_return,
                        'steps': step
                    })
                pbar.update(1)
                # print episode rewards of all agents
                for agent_id in agents.keys():
                    print(f"Agent {agent_id} episode reward: {episode_returns[agent_id]}")
                print(f"All agents episode reward: {sum(episode_returns.values())}")

    return return_dict, episode_returns


class SACAgent:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, obs_dim, act_dim,
                 act_low, act_high,
                 stack_size=4, hidden_size=64, kernel_size=3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, target_entropy=0.0, tau=0.005, gamma=0.99, buffer_size=50000,
                 device="cpu", max_delta=2.5):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.act_low = torch.tensor(act_low, dtype=torch.float32) # absolute action lower bound
        self.act_high = torch.tensor(act_high, dtype=torch.float32) # absolute action upper bound
        # 策略网络
        # self.actor = LSTMPolicyNetContinuous(obs_dim, act_dim, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers).to(device)
        self.actor = PolicyNetContinuous(obs_dim, act_dim, stack_size=stack_size, hidden_size=hidden_size, kernel_size=kernel_size).to(device)
        # 第一个Q网络
        # self.critic_1 = QValueNetContinuous(obs_dim, act_dim, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers).to(device)
        self.critic_1 = QValueNetContinuous(obs_dim, act_dim, stack_size=stack_size, hidden_size=hidden_size, kernel_size=kernel_size).to(device)
        # 第二个Q网络
        # self.critic_2 = QValueNetContinuous(obs_dim, act_dim, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers).to(device)
        self.critic_2 = QValueNetContinuous(obs_dim, act_dim, stack_size=stack_size, hidden_size=hidden_size, kernel_size=kernel_size).to(device)
        self.target_critic_1 = QValueNetContinuous(obs_dim, act_dim, stack_size=stack_size, hidden_size=hidden_size, kernel_size=kernel_size).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(obs_dim, act_dim, stack_size=stack_size, hidden_size=hidden_size, kernel_size=kernel_size).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        # self.actor_hidden = None # for lstm hidden state
        # self.q_hidden = None # for lstm hidden state
        self.action_bound = max_delta

    def take_action(self, state, deterministic: bool = False):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)

        mu, std = self.actor(state)
        if deterministic:
            action = torch.tanh(mu)
        else:
            dist = Normal(mu, std)
            normal_sample = dist.rsample()
            # log_prob = dist.log_prob(normal_sample)
            # 计算tanh_normal分布的对数概率密度
            action = torch.tanh(normal_sample)
            # log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
            
        action = action * self.action_bound

        return action.cpu().detach().numpy().squeeze()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        mu, std = self.actor(next_states)
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        next_log_probs = dist.log_prob(normal_sample)
        next_actions = torch.tanh(normal_sample)
        next_log_probs = next_log_probs - torch.log(1 - torch.tanh(next_actions).pow(2) + 1e-7)
        next_actions = next_actions * self.action_bound

        entropy = -next_log_probs.sum(dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)

        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy.squeeze(-1)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, self.act_dim).to(
            self.device)  
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        mu, std = self.actor(states)
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_probs = dist.log_prob(normal_sample)
        new_actions = torch.tanh(normal_sample)
        log_probs = log_probs - torch.log(1 - torch.tanh(new_actions).pow(2) + 1e-7) # here new_actions is after tanh
        new_actions = new_actions * self.action_bound
        # 直接根据概率计算熵
        entropy = -log_probs.sum(dim=1, keepdim=True)
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        min_qvalue = torch.min(q1_value, q2_value)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if not hasattr(self, "critic_loss_history"):
            self.critic_loss_history = []
        self.critic_loss_history.append(critic_1_loss.item())

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def get_config(self) -> dict:
        """Get agent configuration for saving/loading."""
        return {
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'act_low': self.act_low.tolist(),
            'act_high': self.act_high.tolist(),
            'stack_size': self.stack_size,
            'hidden_size': self.hidden_size,
            'kernel_size': self.kernel_size,
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_1_optimizer.param_groups[0]['lr'],
            'alpha_lr': self.log_alpha_optimizer.param_groups[0]['lr'],
            'target_entropy': self.target_entropy,
            'tau': self.tau,
            'gamma': self.gamma,
            'buffer_size': self.buffer_size,
            'max_delta': self.action_bound,
            'log_alpha': float(self.log_alpha.item()),
        }