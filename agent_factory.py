"""集中管理可訓練的 Agent，供 train.py 與 test.py 共用。"""

from agents.base_agent import BaseAgent
from agents.ddqn_agent import DDQNAgent
from agents.dueling_agent import DuelingAgent
from agents.per_agent import PERAgent
from agents.nstep_per_agent import NStepPERAgent
from agents.noisy_agent import NoisyAgent
from agents.c51_agent import C51Agent
from agents.rainbow_agent import RainbowAgent


AGENT_MAPPING = {
    "VanillaDQN": BaseAgent,
    "DDQN": DDQNAgent,
    "DuelingDQN": DuelingAgent,
    "PER": PERAgent,
    "NStepPER": NStepPERAgent,
    "NoisyNet": NoisyAgent,
    "C51": C51Agent,
    "Rainbow": RainbowAgent,
}


def build_agent(algorithm_name, state_size, action_size, config):
    try:
        agent_class = AGENT_MAPPING[algorithm_name]
    except KeyError as error:
        choices = ", ".join(AGENT_MAPPING)
        raise ValueError(f"不支援的演算法 '{algorithm_name}'。可選：{choices}") from error
    return agent_class(state_size=state_size, action_size=action_size, config=config)
