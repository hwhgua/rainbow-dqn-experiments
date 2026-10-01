import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agents.ddqn_agent import DDQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. C51 專用的網路架構 (輸出機率分佈)
# ==========================================
class C51QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, n_atoms=51, v_min=-300, v_max=300):
        super(C51QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.n_atoms = n_atoms
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # 輸出每個動作在各個 support atom 上的 logits。
        self.fc3 = nn.Linear(fc2_units, action_size * n_atoms)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 將一維的輸出變形為 (Batch, Action, Atoms)
        x = x.view(-1, self.action_size, self.n_atoms)
        # 對 atom 維度套用 Softmax，得到每個動作的機率分佈。
        return F.softmax(x, dim=-1)

# ==========================================
# 2. C51 代理人 (繼承 DDQN)
# ==========================================
class C51Agent(DDQNAgent):
    def __init__(self, state_size, action_size, config):
        super(C51Agent, self).__init__(state_size, action_size, config)
        
        # C51 的超參數設定
        self.n_atoms = getattr(config, 'N_ATOMS', 51)
        self.v_min = getattr(config, 'V_MIN', -300.0)
        self.v_max = getattr(config, 'V_MAX', 300.0)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        
        # 建立固定的 51 個分數刻度 (Support Atoms)
        self.z = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)
        
        # 更換網路為 C51 架構
        self.qnetwork_local = C51QNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS, self.n_atoms, self.v_min, self.v_max).to(device)
        self.qnetwork_target = C51QNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS, self.n_atoms, self.v_min, self.v_max).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)

    def act(self, state, eps=0., deterministic=False):
        """Select an action from expected return values."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            # 單一狀態的輸出維度為 [batch, action_size, n_atoms]。
            action_probs = self.qnetwork_local(state)
        if not deterministic:
            self.qnetwork_local.train()

        if deterministic or np.random.random() > eps:
            # 將機率分佈轉換為期望動作價值：Q(s, a) = sum(p * z)。
            action_values = torch.sum(action_probs * self.z, dim=2)
            return np.argmax(action_values.cpu().numpy())
        return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """執行 C51 機率分佈的 L2 投影與 Cross Entropy Loss 更新"""
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.size(0)

        with torch.no_grad():
            # 1. 決定 Target Network 下一步的動作 (DDQN 邏輯)
            next_action_probs_local = self.qnetwork_local(next_states)
            next_action_values_local = torch.sum(next_action_probs_local * self.z, dim=2)
            best_next_actions = next_action_values_local.argmax(1, keepdim=True)
            
            # 2. 獲取 Target Network 下一步的機率分佈
            next_action_probs_target = self.qnetwork_target(next_states)
            best_next_actions = best_next_actions.unsqueeze(-1).expand(batch_size, 1, self.n_atoms)
            # 抽取最佳動作對應的 51 個機率 (Batch, 51)
            target_probs = next_action_probs_target.gather(1, best_next_actions).squeeze(1)

            # 3. C51 L2 projection。
            # 計算平移後的分數： Tz = R + gamma * Z
            Tz = rewards + (gamma * self.z * (1 - dones))
            Tz = Tz.clamp(min=self.v_min, max=self.v_max) # 防止爆出邊界
            
            # 計算落在浮點數的哪個位置 (例如 index = 4.3)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()  # 下界整數 (4)
            u = b.ceil().long()   # 上界整數 (5)
            
            # 建立一個全 0 的目標機率分佈矩陣
            m = torch.zeros(batch_size, self.n_atoms).to(device)
            
            # 處理邊界條件：如果剛好是一個整數 (l==u)，讓 u = l+1 避免除以 0 的數學錯誤
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) & (l == u)] += 1

            # 將機率依照距離比例分配給下界(l)與上界(u)
            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.n_atoms).to(device)
            # 用 scatter_add_ 進行快速的陣列累加
            m.view(-1).index_add_(0, (l + offset).view(-1), (target_probs * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (target_probs * (b - l.float())).view(-1))

        # 4. 取得 Online Network 目前的機率預測
        current_probs = self.qnetwork_local(states)
        actions_expanded = actions.unsqueeze(-1).expand(batch_size, 1, self.n_atoms)
        current_action_probs = current_probs.gather(1, actions_expanded).squeeze(1)

        # 5. 以 cross-entropy loss 更新分佈價值網路。
        # 為了避免 log(0) 發生，加上一個極小值
        loss = -torch.sum(m * torch.log(current_action_probs + 1e-8), dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_updates += 1

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
