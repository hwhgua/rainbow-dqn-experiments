# agents/ddqn_agent.py
import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent # 引入基礎 Vanilla DQN Agent

class DDQNAgent(BaseAgent):
    """繼承基礎 Agent，僅覆寫 Target Q 的計算邏輯"""
    def __init__(self, state_size, action_size, config):
        # 呼叫父類別 (Vanilla DQN) 的初始化，直接沿用網路與 Buffer
        super(DDQNAgent, self).__init__(state_size, action_size, config)
        
    def learn(self, experiences, gamma):
        """只改寫 learn 函數，實作 Double DQN 邏輯"""
        states, actions, rewards, next_states, dones = experiences

        # --- DDQN 核心修改 ---
        best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions).detach()
        # ---------------------
        
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_updates += 1

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
