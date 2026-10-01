import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agents.ddqn_agent import DDQNAgent # 確保實驗對比乾淨，繼承自純 DDQN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 自定義的雜訊線性層 (NoisyLinear)
# ==========================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # --- 可學習參數 (Learnable Parameters) ---
        # 這些參數會接收反向傳播的梯度更新
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # --- 非學習參數 (Buffers) ---
        # 儲存隨機抽樣的 epsilon 雜訊。
        # 使用 register_buffer 是因為它們屬於模組狀態，但不參與梯度計算 (requires_grad=False)。
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        初始化 Mu 和 Sigma。
        依據論文，初始化範圍與輸入特徵數 (in_features) 的平方根成反比，以確保訓練初期的穩定性。
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """
        產生 Factorized Gaussian Noise (因式分解常態雜訊)。
        數學轉換：f(x) = sgn(x) * sqrt(|x|)
        目的：減少計算量。比起直接生成 M*N 個雜訊，只生成 M 個與 N 個雜訊再做外積，效能更高。
        """
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """
        重新抽樣雜訊。
        在每一次選擇動作 (act) 或進行學習 (learn) 之前呼叫。
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # 使用外積 (Outer Product) 生成權重雜訊矩陣，並直接賦值給 bias_epsilon
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """
        前向傳播。
        僅在 model.train() 模式下加入雜訊；在 model.eval() 模式下（如純測試時）僅使用 mu 進行決策。
        """
        if self.training:
            # 依據公式：W = mu_w + sigma_w * eps_w
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)


# ==========================================
# 2. 替換為 Noisy Linear 的 Q-Network
# ==========================================
class NoisyQNetwork(nn.Module):
    """
    架構：輸入層 -> 隱藏層 (標準) -> 隱藏層 (Noisy) -> 輸出層 (Noisy)
    實務上，第一層輸入層通常保留標準 nn.Linear，以避免對原始狀態特徵產生過度破壞。
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(NoisyQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = NoisyLinear(fc1_units, fc2_units)
        self.fc3 = NoisyLinear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
    def reset_noise(self):
        """重新抽樣網路中所有 NoisyLinear 層的雜訊。"""
        self.fc2.reset_noise()
        self.fc3.reset_noise()


# ==========================================
# 3. 實作 Noisy Agent (繼承自 DDQN)
# ==========================================
class NoisyAgent(DDQNAgent):
    """
    嚴謹對照組：純粹測試 Noisy Nets 取代 Epsilon 的代理人。
    控制變數：排除 PER 與 N-Step 的干擾，僅繼承 DDQN 的 Double Q-Learning 邏輯。
    """
    exploration_strategy = "noisy_network"
    uses_epsilon_greedy = False

    def __init__(self, state_size, action_size, config):
        # 呼叫 DDQN 初始化，建立標準的 ReplayBuffer
        super(NoisyAgent, self).__init__(state_size, action_size, config)
        
        # 覆寫網路：將原本的 QNetwork 替換為帶有雜訊的 NoisyQNetwork
        self.qnetwork_local = NoisyQNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS).to(device)
        self.qnetwork_target = NoisyQNetwork(state_size, action_size, config.SEED, config.FC1_UNITS, config.FC2_UNITS).to(device)
        
        # 網路物件已更換，必須重新將新網路的參數綁定給優化器 (Optimizer)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)

    def act(self, state, eps=0., deterministic=False):
        """Use NoisyNet exploration for training and a noise-free policy for evaluation."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if deterministic:
            # NoisyLinear.eval() uses mu only, so no parameter noise is added.
            self.qnetwork_local.eval()
        else:
            self.qnetwork_local.train()
            self.qnetwork_local.reset_noise()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        return np.argmax(action_values.cpu().numpy())

    def learn(self, experiences, gamma):
        """
        在計算 Target Q 與 Expected Q 前重新抽樣雜訊，再執行 DDQN 更新。
        """
        self.qnetwork_local.train()
        self.qnetwork_target.train()
        # 依據論文規範：在每次梯度更新前，Online 網路與 Target 網路都必須重新抽樣雜訊
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
        
        # 呼叫父類別 (DDQNAgent) 的 learn 方法執行 Double Q-Learning 更新
        super(NoisyAgent, self).learn(experiences, gamma)
