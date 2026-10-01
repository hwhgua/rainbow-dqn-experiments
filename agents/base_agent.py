import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import Config
from buffer import ReplayBuffer

# 決定使用 CPU 或 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """基礎 Q-Network：將 ``state_size`` 維狀態映射為 ``action_size`` 個動作價值。"""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):  # 三層全連接網路
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)  # 固定 PyTorch 的隨機種子
        self.fc1 = nn.Linear(state_size, fc1_units)  # state_size -> fc1_units
        self.fc2 = nn.Linear(fc1_units, fc2_units)   # fc1_units -> fc2_units
        self.fc3 = nn.Linear(fc2_units, action_size) # fc2_units -> action_size

    def forward(self, state):  # 向前使用的激活函數使用relu
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class BaseAgent():
    exploration_strategy = "epsilon_greedy"
    uses_epsilon_greedy = True

    def __init__(self, state_size, action_size, config: Config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config 
        
        # 將訓練設定保留為實例屬性，供更新流程使用。
        self.batch_size = config.BATCH_SIZE
        self.update_every = config.UPDATE_EVERY
        self.gamma = config.GAMMA
        self.tau = config.TAU

        self.qnetwork_local = QNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)
        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, config.SEED)
        self.t_step = 0
        # Exposed in run summaries so an experiment can distinguish a merely
        # executable rollout from a run that actually performed optimization.
        self.gradient_updates = 0
    
    def step(self, state, action, reward, next_state, done, episode_end=None):
        """Store a transition and decide whether to learn.

        ``done`` is the Bellman terminal mask. ``episode_end`` is accepted for
        a common Agent interface; standard replay does not need it separately.
        """
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0., deterministic=False):
        """Return an epsilon-greedy action or a deterministic greedy action."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if not deterministic:
            self.qnetwork_local.train()

        if deterministic or random.random() > eps:
            return np.argmax(action_values.cpu().numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """使用 Batch 經驗更新價值網路"""
        states, actions, rewards, next_states, dones = experiences

        # 計算 Target Q
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 取得 Expected Q (Online 網路的預測值)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 計算 Loss 並優化 (MSE Loss)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_updates += 1

        # 更新 Target Network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update 模型參數: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
