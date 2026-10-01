# agents/dueling_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from agents.ddqn_agent import DDQNAgent # 繼承 DDQN 的 learn() 邏輯

# 必須先定義 Dueling 特有的神經網路結構
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # 1. 共通特徵層 (Feature Layer)：萃取環境狀態的基礎特徵
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU()
        )

        # 2. Value Stream (狀態價值 V)：評估這個狀態本身有多安全/多好，輸出為純量 (1 維)
        self.value_stream = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1) 
        )

        # 3. Advantage Stream（優勢函數 A）：輸出 action_size 個動作的相對優勢。
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size) 
        )

    def forward(self, state):
        features = self.feature_layer(state)
        
        # 兩條分支各自計算
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        
        # 最終合併公式：Q(s,a) = V(s) + (A(s,a) - mean(A))
        # 使用 A.mean() 是為了確保網路的穩定性 (Identifiability)
        return V + (A - A.mean(dim=1, keepdim=True))


class DuelingAgent(DDQNAgent):
    """
    以 Dueling 網路取代 DDQN 的標準 Q-Network。
    """
    def __init__(self, state_size, action_size, config):
        super(DuelingAgent, self).__init__(state_size, action_size, config)
        
        # 重新建立 local 與 target 網路，採用 DuelingQNetwork。
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS).to(device)
        
        # 網路更換後，重新建立對應參數的優化器。
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)
