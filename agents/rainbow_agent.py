# agents/rainbow_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from agents.base_agent import BaseAgent
from agents.noisy_agent import NoisyLinear
from buffer import PrioritizedReplayBuffer, NStepDeque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 融合網路：Dueling + Noisy + C51
# ==========================================
class RainbowQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, n_atoms=51, fc1_units=64, fc2_units=64):
        super(RainbowQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.n_atoms = n_atoms
        
        # 共用特徵層 (標準 Linear)
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        # Dueling 架構的 advantage stream，使用 NoisyLinear。
        # 優勢流 (Advantage)
        self.advantage_fc = NoisyLinear(fc1_units, fc2_units)
        self.advantage_out = NoisyLinear(fc2_units, action_size * n_atoms)
        
        # 價值流 (Value)
        self.value_fc = NoisyLinear(fc1_units, fc2_units)
        self.value_out = NoisyLinear(fc2_units, 1 * n_atoms)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        
        # 計算優勢與價值，並將維度展開為 (Batch, Action, Atoms)
        a = F.relu(self.advantage_fc(x))
        a = self.advantage_out(a).view(-1, self.action_size, self.n_atoms)
        
        v = F.relu(self.value_fc(x))
        v = self.value_out(v).view(-1, 1, self.n_atoms)
        
        # 結合 V 與 A，並使用 Softmax 輸出機率分佈 (C51)
        q_dist = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(q_dist, dim=-1)

    def reset_noise(self):
        """重新抽樣網路中所有 NoisyLinear 層的雜訊。"""
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
        self.value_fc.reset_noise()
        self.value_out.reset_noise()


# ==========================================
# 融合代理人：Rainbow Agent
# ==========================================
class RainbowAgent(BaseAgent):
    exploration_strategy = "noisy_network"
    uses_epsilon_greedy = False

    def __init__(self, state_size, action_size, config):
        super(RainbowAgent, self).__init__(state_size, action_size, config)
        
        # --- C51 超參數 ---
        self.n_atoms = getattr(config, 'N_ATOMS', 51)
        self.v_min = getattr(config, 'V_MIN', -200.0) # C51 support 的下界。
        self.v_max = getattr(config, 'V_MAX', 300.0)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)
        
        # --- N-Step 與 PER 超參數 ---
        self.n_step = getattr(config, 'N_STEP', 3)
        self.beta_start = config.PER_BETA_START
        self.beta_end = config.PER_BETA_END
        self.beta = self.beta_start
        self.beta_anneal_steps = config.PER_BETA_ANNEAL_STEPS
        self.beta_step = 0
        self.beta_increment = (self.beta_end - self.beta_start) / config.N_EPISODES

        # 初始化 N-step transition buffer 與 PER replay buffer。
        self.n_step_buffer = NStepDeque(self.n_step, self.gamma)
        self.memory = PrioritizedReplayBuffer(
            action_size, config.BUFFER_SIZE, config.BATCH_SIZE, config.SEED
        )

        # 初始化 Rainbow 的 local 與 target 網路。
        self.qnetwork_local = RainbowQNetwork(
            state_size,
            action_size,
            config.SEED,
            self.n_atoms,
            config.FC1_UNITS,
            config.FC2_UNITS,
        ).to(device)
        self.qnetwork_target = RainbowQNetwork(
            state_size,
            action_size,
            config.SEED,
            self.n_atoms,
            config.FC1_UNITS,
            config.FC2_UNITS,
        ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)

    def _advance_beta(self, episode_end):
        if self.beta_anneal_steps is not None:
            self.beta_step += 1
            progress = min(1.0, self.beta_step / self.beta_anneal_steps)
            self.beta = self.beta_start + progress * (self.beta_end - self.beta_start)
        elif episode_end:
            self.beta = min(self.beta_end, self.beta + self.beta_increment)

    def step(self, state, action, reward, next_state, done, episode_end=None):
        """資料管線融合"""
        if episode_end is None:
            episode_end = done
        # 1. 經過 N-Step 攔截器打包
        n_step_experiences = self.n_step_buffer.append(
            state, action, reward, next_state, done, episode_end=episode_end
        )
        
        # 2. 存入 PER Buffer
        for n_state, n_action, n_reward, n_next_state, n_done, n_steps in n_step_experiences:
            self.memory.add(n_state, n_action, n_reward, n_next_state, n_done, n_steps=n_steps)
        
        # 3. 觸發學習
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                # 從 PER buffer 取得樣本、樹節點索引與 IS weights。
                experiences, indices, is_weights = self.memory.sample(self.beta)
                self.learn(experiences, indices, is_weights, self.gamma)
                
        self._advance_beta(episode_end)

    def act(self, state, eps=0., deterministic=False):
        """Use NoisyNet exploration for training and a noise-free policy for evaluation."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if deterministic:
            self.qnetwork_local.eval()
        else:
            self.qnetwork_local.train()
            self.qnetwork_local.reset_noise()
        
        with torch.no_grad():
            action_probs = self.qnetwork_local(state)
            
        # 將機率分佈轉為期望值: sum(p * z)
        action_values = torch.sum(action_probs * self.z, dim=2)
        
        # 選擇期望動作價值最高的動作。
        return np.argmax(action_values.cpu().numpy())

    def learn(self, experiences, indices, is_weights, gamma):
        """以 Double DQN、C51、N-step 與 PER 執行 Rainbow 更新。"""
        states, actions, rewards, next_states, dones, n_steps = experiences
        batch_size = states.size(0)

        self.qnetwork_local.train()
        self.qnetwork_target.train()
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        with torch.no_grad():
            # Double DQN：由 local 網路選擇下一個動作。
            next_action_probs_local = self.qnetwork_local(next_states)
            next_action_values_local = torch.sum(next_action_probs_local * self.z, dim=2)
            best_next_actions = next_action_values_local.argmax(1, keepdim=True)
            
            # 用 Target 網路評估該動作的機率分佈
            next_action_probs_target = self.qnetwork_target(next_states)
            best_next_actions = best_next_actions.unsqueeze(-1).expand(batch_size, 1, self.n_atoms)
            target_probs = next_action_probs_target.gather(1, best_next_actions).squeeze(1)

            # C51 projection；尾端短序列使用其實際跨越的步數。
            gamma_n = torch.pow(torch.full_like(n_steps, gamma), n_steps)
            Tz = rewards + (gamma_n * self.z * (1 - dones))
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            m = torch.zeros(batch_size, self.n_atoms).to(device)
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) & (l == u)] += 1

            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.n_atoms).to(device)
            m.view(-1).index_add_(0, (l + offset).view(-1), (target_probs * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (target_probs * (b - l.float())).view(-1))

        # 取得當前預測分佈
        current_probs = self.qnetwork_local(states)
        actions_expanded = actions.unsqueeze(-1).expand(batch_size, 1, self.n_atoms)
        current_action_probs = current_probs.gather(1, actions_expanded).squeeze(1)

        # 更新 PER 優先權，並計算 importance-sampling 加權 loss。
        # 1. 計算每個樣本的 Cross Entropy (這就是 C51 的 TD Error)
        elementwise_loss = -torch.sum(m * torch.log(current_action_probs + 1e-8), dim=1)
        
        # 2. 將誤差回傳給 SumTree 更新優先權 (Loss 越大越優先被抽中)
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # 3. 乘上 IS_Weights 修正抽樣偏差，並取平均作為最終 Loss
        is_weights = is_weights.unsqueeze(1)
        loss = (is_weights.squeeze() * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # 實務建議：加上梯度裁剪 (Gradient Clipping) 穩定訓練
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_updates += 1

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
