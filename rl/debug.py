import gym
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque

# --------------------------------------------------------------------------------
# 1. 一些超参数的定义：可根据实际需求在此处调整
# --------------------------------------------------------------------------------
SEED = 42                    # 随机种子
TOTAL_TIMESTEPS = 200_000    # 训练环境交互步数
INITIAL_RANDOM_STEPS = 10_000  # 初期随机采样步数
BATCH_SIZE = 256
LR_ENCODER = 1e-4
LR_QPOLICY = 3e-4
GAMMA = 0.99
BUFFER_CAPACITY = 1_000_000
TARGET_UPDATE_FREQ = 250     # 多少步同步一次target网络
NSTEP_RETURN = 3             # 多步序列用于价值回报
UNROLL_STEPS = 5             # 在encoder中对模型进行unroll的步数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------
# 2. 经验回放池(Replay Buffer)
#   存储 (s, a, r, d, s') 元组，以及为多步回报准备的队列/指针
# --------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        # 从buffer中随机采样一个批次
        batch = random.sample(self.buffer, batch_size)
        s, a, r, d, s_next = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(a), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(r), dtype=torch.float32, device=DEVICE).unsqueeze(-1),
            torch.tensor(np.array(d), dtype=torch.float32, device=DEVICE).unsqueeze(-1),
            torch.tensor(np.array(s_next), dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------------
# 3. 状态编码器(可根据观测的类型使用CNN或MLP)
#   - f(s): 将状态s映射到latent向量 zs
# --------------------------------------------------------------------------------
class StateEncoder(nn.Module):
    def __init__(self, state_dim=24, latent_dim=512, is_image=False):
        super().__init__()
        self.is_image = is_image
        self.latent_dim = latent_dim
        if is_image:
            # 如果状态是图像，可用简单CNN作为encoder
            # 示例：假设状态是[84x84灰度图] => (batch_size, 1, 84, 84)
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2),
                nn.ReLU(),
            )
            # 卷积后需要flatten成为向量
            self.fc = nn.Linear(32 * 9 * 9, latent_dim)
        else:
            # 如果是向量观察，直接用多层感知器(MLP)
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim),
            )

    def forward(self, s):
        # 输入s形状可能不同，若是图像需先通道对齐
        if self.is_image:
            # 这里示例假设已经reshape成 (batch, 1, 84, 84) 并是0-1的float
            x = self.conv(s)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.fc(s)
        return x


# --------------------------------------------------------------------------------
# 4. 状态-动作编码器 g(zs, a) => zsa
#   - 输入：zs(上面得到的状态潜在向量)，以及动作a
#   - 输出：zsa(状态-动作的混合潜在表示)
# --------------------------------------------------------------------------------
class StateActionEncoder(nn.Module):
    def __init__(self, zs_dim=512, action_dim=6, zsa_dim=512):
        super().__init__()
        self.zsa_dim = zsa_dim
        self.zs_dim = zs_dim
        self.action_dim = action_dim

        self.action_fc = nn.Linear(action_dim, 256)  # 将动作投影到一个中间维度
        self.fc_net = nn.Sequential(
            nn.Linear(zs_dim + 256, 512),
            nn.ReLU(),
            nn.Linear(512, zsa_dim),
        )

        # 用于预测 「下一步状态embedding / 奖励 / 终止」的线性层
        # output_dim = zs_dim(预测下一个状态embedding) + 1(预测reward) + 1(预测done)
        self.model_head = nn.Linear(zsa_dim, zs_dim + 1 + 1)

    def forward(self, zs, a):
        a_proj = F.relu(self.action_fc(a))
        x = torch.cat([zs, a_proj], dim=-1)
        zsa = self.fc_net(x)
        # 后面可再做线性映射得到对下个状态(zs'), reward, done的预测
        pred = self.model_head(zsa)
        return zsa, pred


# --------------------------------------------------------------------------------
# 5. 价值网络 Q(zsa)
#   - 这里我们以TD3思路，需要两个Q网络(取它们的min作目标)
# --------------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, zsa_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zsa_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, zsa):
        return self.net(zsa)


# --------------------------------------------------------------------------------
# 6. 策略网络 π(zs) => a
#   - 连续动作: 输出通常通过tanh或clip到 [-1, 1]
#   - 离散动作: 可用Gumbel-Softmax或直接输出分类
#   这里以连续动作为例
# --------------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, zs_dim=512, action_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, zs):
        # 输出一个未激活的向量，然后做tanh
        return torch.tanh(self.net(zs))


# --------------------------------------------------------------------------------
# 7. MR.Q 主体类
#   - 包含:
#       (1) encoder相关模块 (f, g, model对reward/下个状态embedding/终止的预测)
#       (2) 两个Q网络 + 目标网络
#       (3) policy网络 + 目标网络
#       (4) 训练逻辑: encoder loss + Q更新 + policy更新 + target网络同步
# --------------------------------------------------------------------------------
class MRQAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 is_image=False,
                 latent_dim=512,
                 zsa_dim=512,
                 device=DEVICE):

        self.device = device

        # 状态编码器 & 状态-动作编码器
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            is_image=is_image
        ).to(device)

        self.state_action_encoder = StateActionEncoder(
            zs_dim=latent_dim,
            action_dim=action_dim,
            zsa_dim=zsa_dim
        ).to(device)

        # 两个Q网络 + target
        self.Q1 = QNetwork(zsa_dim=zsa_dim).to(device)
        self.Q2 = QNetwork(zsa_dim=zsa_dim).to(device)
        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q2_target = copy.deepcopy(self.Q2)

        # 策略网络 + target
        self.policy = PolicyNetwork(zs_dim=latent_dim, action_dim=action_dim).to(device)
        self.policy_target = copy.deepcopy(self.policy)

        # 优化器：encoder和model一起；Q和policy一起或分开
        self.encoder_optimizer = torch.optim.AdamW(
            list(self.state_encoder.parameters()) +
            list(self.state_action_encoder.parameters()),
            lr=LR_ENCODER
        )
        self.qpolicy_optimizer = torch.optim.AdamW(
            list(self.Q1.parameters()) +
            list(self.Q2.parameters()) +
            list(self.policy.parameters()),
            lr=LR_QPOLICY
        )

        # 用于统计平均 reward 以作可选的reward scale
        self.avg_abs_reward = 1.0

        self.total_updates = 0  # 用于判断何时同步target

    def update(self, replay_buffer):

        # 1. 从回放池中采样
        s, a, r, d, s_next = replay_buffer.sample(BATCH_SIZE)

        # 2. 获取当前和下一个状态的embedding
        zs = self.state_encoder(s)       # 维度: [batch_size, latent_dim]
        zs_next = self.state_encoder(s_next).detach()  # target encoder等价:先不做梯度

        # (可选) 平滑averge abs reward，用于normalize
        with torch.no_grad():
            self.avg_abs_reward = 0.99 * self.avg_abs_reward + 0.01 * r.abs().mean().item()

        # 3. encoder部分: 模型式loss
        #   利用 unroll 步数, 去预测后续的 zs, r, d（这里仅演示简单1步或多步）
        #   这里为了示范，只写1个循环，真正实现时可叠加UNROLL_STEPS
        loss_model = 0.0
        z_current = zs
        for t in range(UNROLL_STEPS):
            # 状态-动作 -> 预测(r, done, next_z)
            zsa, pred = self.state_action_encoder(z_current, a)
            # model_head 分解
            pred_z_next = pred[:, :zs_next.shape[1]]
            pred_r      = pred[:, zs_next.shape[1] : zs_next.shape[1]+1]
            pred_done   = pred[:, zs_next.shape[1]+1 : zs_next.shape[1]+2]

            # 与实际标签的差
            # 这里简单用MSE( reward也用MSE演示, 文章里还可以做分类预测 )
            # done预测也用MSE
            loss_r = F.mse_loss(pred_r, r / self.avg_abs_reward)
            loss_d = F.mse_loss(pred_done, d)
            # 动力学loss
            with torch.no_grad():
                # next_z_target(对s_next做编码)；如果是多步 unroll, 需要再从 buffer 取后续状态
                # 这里为了简单都用 s_next
                next_z_target = self.state_encoder(s_next).detach()
            loss_dyn = F.mse_loss(pred_z_next, next_z_target)

            loss_model = loss_model + (1.0 * loss_r + 0.1 * loss_d + 1.0 * loss_dyn)

            # 这里忽略多步中后续 s 的获取过程，仅做示范
            # z_current 替换成 pred_z_next 以模拟滚动
            z_current = pred_z_next.detach()

        loss_model = loss_model / UNROLL_STEPS

        # 4. Q更新: 计算 TD目标 = n-step return + γ Q'(z_{s'})   (简化写法)
        with torch.no_grad():
            # next_action = policy_target(zs_next) + 高斯噪声(若TD3式target需要)
            next_action = self.policy_target(zs_next)
            zsa_next, _ = self.state_action_encoder(zs_next, next_action)
            Q1_next = self.Q1_target(zsa_next)
            Q2_next = self.Q2_target(zsa_next)
            Q_next = torch.min(Q1_next, Q2_next)
            # TD target
            target_Q = r / self.avg_abs_reward + (1 - d) * GAMMA * Q_next

        # 计算Q1, Q2的当前预测
        zsa_cur, _ = self.state_action_encoder(zs, a)
        Q1_cur = self.Q1(zsa_cur)
        Q2_cur = self.Q2(zsa_cur)
        loss_q1 = F.smooth_l1_loss(Q1_cur, target_Q)
        loss_q2 = F.smooth_l1_loss(Q2_cur, target_Q)
        loss_q = loss_q1 + loss_q2

        # 5. Policy更新: DPG，最大化Q1 -> 最小化 - Q1
        #    注意：此处让policy的输入是 zs, 我们只算 第一个Q1的梯度
        predict_action = self.policy(zs)
        zsa_pol, _ = self.state_action_encoder(zs, predict_action)
        policy_loss = -self.Q1(zsa_pol).mean()

        # 6. 总体loss = encoder(模型式) + Q部分 + policy部分
        #   一般会分两个optimizer或多步更新，这里为演示放一起
        #   也可考虑用不同权重balance，这里简单相加
        loss = loss_model + loss_q + policy_loss

        self.encoder_optimizer.zero_grad()
        self.qpolicy_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.qpolicy_optimizer.step()

        self.total_updates += 1
        # 若到达target同步频率，则更新target网络
        if self.total_updates % TARGET_UPDATE_FREQ == 0:
            self.Q1_target.load_state_dict(self.Q1.state_dict())
            self.Q2_target.load_state_dict(self.Q2.state_dict())
            self.policy_target.load_state_dict(self.policy.state_dict())

        return {
            "loss_model": loss_model.item(),
            "loss_q": loss_q.item(),
            "loss_policy": policy_loss.item()
        }

    def select_action(self, s, noise_scale=0.2, train=True):
        # 输入原始状态 s (numpy数组)
        self.state_encoder.eval()
        self.policy.eval()

        with torch.no_grad():
            s_tensor = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            zs = self.state_encoder(s_tensor)
            a = self.policy(zs)
            action = a.cpu().numpy()[0]
        self.state_encoder.train()
        self.policy.train()

        # 训练时加噪音以探索；若是离散动作则需做其他处理
        if train:
            action = action + np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action, -1.0, 1.0)


# --------------------------------------------------------------------------------
# 8. 主训练流程: 以一个gym的连续动作任务为例 (比如"HalfCheetah-v4")
# --------------------------------------------------------------------------------
def main():
    env_name = "HalfCheetah-v4"  # 或者替换成你的环境，比如 "Walker2d-v4"
    env = gym.make(env_name)
    env.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作

    agent = MRQAgent(state_dim=state_dim, action_dim=action_dim, is_image=False)

    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # 用于记录一些简单的训练统计
    episode_reward = 0
    episode = 0
    state = env.reset()
    for t in range(TOTAL_TIMESTEPS):
        # 1. 初期随机动作
        if t < INITIAL_RANDOM_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise_scale=0.2, train=True)

        next_state, reward, done, info = env.step(action)
        replay_buffer.push((state, action, reward, float(done), next_state))

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            print(f"Episode {episode}, total_reward = {episode_reward:.2f}")
            episode_reward = 0
            episode += 1

        # 2. 当buffer中有足够多数据时，开始训练update
        if len(replay_buffer) > BATCH_SIZE * 5:
            losses = agent.update(replay_buffer)

        # 3. 仅作演示，若需要评估可在此插入eval逻辑
        # ...

    env.close()
    print("训练完成")


if __name__ == "__main__":
    main()