# buffer.py
import numpy as np
import random
import torch
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 標準經驗回放池 (向下相容保留給 Vanilla, DDQN, Dueling)
# ==========================================
class ReplayBuffer:
    """固定大小的標準緩衝區，使用均勻隨機抽樣"""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    
# ==========================================
# 2. SumTree 資料結構 (實作 O(log N) 的更新與搜尋)
# ==========================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # 樹的總節點數為 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # 用一個獨立的陣列存實際的經驗資料 (state, action...)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """遞迴更新父節點"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """更新某個葉節點的優先權，並將差異向上傳遞"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        """新增一筆經驗到樹的葉節點"""
        # 葉節點的起始 index 是 capacity - 1
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0 # 滿了就從頭覆蓋
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        """向下搜尋，找出落點對應的葉節點"""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # 如果已經沒有子節點，代表已經到達葉節點
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx # 往左走
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx # 往右走

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0] # 根節點即為總和


# ==========================================
# 3. 優先經驗回放池 (PER Buffer)
# ==========================================
class PrioritizedReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6):
        self.action_size = action_size
        self.batch_size = batch_size
        self.alpha = alpha # 優先權的程度 (0 = 純隨機, 1 = 完全看優先權)
        self.tree = SumTree(buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done", "n_steps"],
        )
        self.seed = random.seed(seed)
        self.epsilon = 1e-5 # 小常數，避免 TD Error 為 0 時永遠不被抽中

    def add(self, state, action, reward, next_state, done, n_steps=1):
        e = self.experience(state, action, reward, next_state, done, n_steps)
        # 關鍵邏輯：新加入的經驗，直接給予當前樹中的最大優先權，確保它至少被抽中學習一次
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, e)

    def sample(self, beta=0.4):
        experiences = []
        indices = []
        priorities = []
        
        # 將總機率切成 batch_size 個區段，每個區段抽一個，確保抽樣的均勻性
        segment = self.tree.total_priority / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get_leaf(s)
            indices.append(idx)
            priorities.append(priority)
            experiences.append(data)
            
        # 計算 Importance Sampling (IS) Weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max() # 為了訓練穩定，將權重正規化到 0~1 之間
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        n_steps = torch.from_numpy(np.vstack([e.n_steps for e in experiences if e is not None])).float().to(device)
  
        # 回傳資料時，多回傳了 indices (才知道等一下要更新誰) 與 IS weights (修正偏誤)
        return (states, actions, rewards, next_states, dones, n_steps), indices, torch.from_numpy(is_weights).float().to(device)

    def update_priorities(self, indices, td_errors):
        """根據神經網路算出的 TD Error 更新樹中的優先權"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            
    def __len__(self):
        return self.tree.n_entries

# ==========================================
# 4. N 步資料攔截器 (N-Step Deque)
# ==========================================
class NStepDeque:
    """用來打包 N-step 累積獎勵的暫存佇列"""
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=n_step)

    def append(self, state, action, reward, next_state, done, episode_end=None):
        """回傳本次可寫入 replay buffer 的一或多筆 N-step 經驗。

        ``done`` 是 Bellman target 的真正 terminal mask；``episode_end``
        則同時涵蓋 terminated 與 truncated，用來在每個 episode 邊界
        flush 剩餘短序列，避免跨 episode 拼接 transition。
        """
        if episode_end is None:
            episode_end = done
        self.memory.append((state, action, reward, next_state, done))

        completed = []
        if len(self.memory) == self.n_step:
            completed.append(self._get_n_step_info())
            self.memory.popleft()

        if episode_end:
            while self.memory:
                completed.append(self._get_n_step_info())
                self.memory.popleft()

        return completed

    def _get_n_step_info(self):
        """計算 N 步累積獎勵與對應的首尾狀態"""
        # 起點狀態與動作
        state_0, action_0, _, _, _ = self.memory[0]
        # 終點狀態與結束訊號
        _, _, _, next_state_n, done_n = self.memory[-1]

        # 核心數學：計算折扣累積獎勵 R_t + γ*R_{t+1} + γ^2*R_{t+2} ...
        n_reward = 0
        for i in range(len(self.memory)):
            n_reward += self.memory[i][2] * (self.gamma ** i)

        return state_0, action_0, n_reward, next_state_n, done_n, len(self.memory)
