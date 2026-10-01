# agents/nstep_per_agent.py
import torch
from agents.per_agent import PERAgent
from buffer import NStepDeque

class NStepPERAgent(PERAgent):
    """結合 N-Step Learning 與 PER 的代理人"""
    def __init__(self, state_size, action_size, config):
        # 呼叫 PERAgent 的初始化 (這會把 PER Buffer 跟 Beta 參數都設好)
        super(NStepPERAgent, self).__init__(state_size, action_size, config)
        
        # 新增 N-step 參數
        self.n_step = getattr(config, 'N_STEP', 3) # 預設為 3
        # 建立 N 步資料攔截器
        self.n_step_buffer = NStepDeque(self.n_step, self.gamma)

    def step(self, state, action, reward, next_state, done, episode_end=None):
        """覆寫 step: 先經過 N 步攔截器打包，再存入 PER Buffer"""
        if episode_end is None:
            episode_end = done
        
        # 1. 餵給攔截器
        n_step_experiences = self.n_step_buffer.append(
            state, action, reward, next_state, done, episode_end=episode_end
        )
        
        # 2. 如果攔截器吐出了打包好的經驗，才存入真正的 PER Buffer
        for n_state, n_action, n_reward, n_next_state, n_done, n_steps in n_step_experiences:
            self.memory.add(n_state, n_action, n_reward, n_next_state, n_done, n_steps=n_steps)
        
        # 以下為原本的訓練觸發邏輯
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences, indices, is_weights = self.memory.sample(self.beta)
                self.learn(experiences, indices, is_weights, self.gamma)
                
        self._advance_beta(episode_end)

    def learn(self, experiences, indices, is_weights, gamma):
        """覆寫 learn: 修正 Bellman Equation 中的 gamma 次方"""
        states, actions, rewards, next_states, dones, n_steps = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 尾端短序列（特別是 truncated）不一定剛好有 N 步，因此每筆
        # transition 都使用自己實際跨越的折扣步數。
        gamma_n = torch.pow(torch.full_like(n_steps, gamma), n_steps)
        Q_targets = rewards + (gamma_n * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 計算 TD Error 並更新 PER 樹狀結構
        td_errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors)

        # 計算加權 MSE Loss
        is_weights = is_weights.unsqueeze(1)
        loss = (is_weights * (Q_expected - Q_targets) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_updates += 1

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
