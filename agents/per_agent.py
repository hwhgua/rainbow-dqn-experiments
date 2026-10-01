# agents/per_agent.py
import torch
import numpy as np
from agents.base_agent import BaseAgent
from buffer import PrioritizedReplayBuffer

class PERAgent(BaseAgent):
    """加入優先經驗回放 (PER) 的代理人"""
    def __init__(self, state_size, action_size, config):
        # 呼叫父類別初始化網路
        super(PERAgent, self).__init__(state_size, action_size, config)
        
        # 以 PrioritizedReplayBuffer 取代標準 replay buffer。
        self.memory = PrioritizedReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, config.SEED)
        
        # The legacy interactive path anneals beta per episode.  Controlled
        # experiments provide PER_BETA_ANNEAL_STEPS so all agents advance it
        # on the same environment-step schedule instead.
        self.beta_start = config.PER_BETA_START
        self.beta_end = config.PER_BETA_END
        self.beta = self.beta_start
        self.beta_anneal_steps = config.PER_BETA_ANNEAL_STEPS
        self.beta_step = 0
        self.beta_increment = (self.beta_end - self.beta_start) / config.N_EPISODES

    def _advance_beta(self, episode_end):
        if self.beta_anneal_steps is not None:
            self.beta_step += 1
            progress = min(1.0, self.beta_step / self.beta_anneal_steps)
            self.beta = self.beta_start + progress * (self.beta_end - self.beta_start)
        elif episode_end:
            self.beta = min(self.beta_end, self.beta + self.beta_increment)

    def step(self, state, action, reward, next_state, done, episode_end=None):
        """覆寫 step: 處理抽樣與 Beta 增長"""
        if episode_end is None:
            episode_end = done
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                # PER 抽樣會回傳樹節點索引與 importance-sampling weights。
                experiences, indices, is_weights = self.memory.sample(self.beta)
                self.learn(experiences, indices, is_weights, self.gamma)
                
        self._advance_beta(episode_end)

    def learn(self, experiences, indices, is_weights, gamma):
        """覆寫 learn: 實作加權 Loss 與更新樹狀結構"""
        states, actions, rewards, next_states, dones, _ = experiences

        # 計算 Target Q (這裡先用 Vanilla DQN 的算法作示範)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 取得 Expected Q
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 依 TD error 更新 SumTree 優先權。
        # 把 Tensor 轉回 numpy 陣列才能傳給 Buffer
        td_errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors)

        # 將 is_weights 從 (64,) 擴張成 (64, 1) 來對齊 Q 值的維度
        is_weights = is_weights.unsqueeze(1)

        # 以 importance-sampling weights 計算加權 MSE loss。
        # 不能直接用原本的 F.mse_loss，因為它會自動把整批誤差平均掉。
        # 因此先計算逐樣本平方誤差，乘上 is_weights 後再取平均。
        loss = (is_weights * (Q_expected - Q_targets) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_updates += 1

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
