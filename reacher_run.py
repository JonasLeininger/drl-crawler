import numpy as np

from replay_buffer import ReplayBuffer
from config.config import Config
from agents.ddpg_agent import DDPGAgent

def main():
    config = Config()
    buffer_size = int(1e5)
    replay = ReplayBuffer(buffer_size, 1)
    print_env_information(config)
    print(config.state_dim)
    agent = DDPGAgent(config)
    train_agent(config, agent)
    config.env.close()


def print_env_information(config):
    config.env_info = config.env.reset(train_mode=False)[config.brain_name]
    config.num_agents = len(config.env_info.agents)
    print('Number of agents:', config.num_agents)
    print('Size of each action:', config.action_dim)
    config.states = config.env_info.vector_observations
    print('There are {} agents. Each observes a state with length: {}'.format(config.states.shape[0], config.state_dim))
    print('The state for the first agent looks like:', config.states[0])


def run_random_env(config, replay):
    env_info = config.env.reset(train_mode=False)[config.brain_name]
    states = env_info.vector_observations
    scores = np.zeros(config.num_agents)
    steps = 1000
    for t in range(steps):
        actions = np.random.randn(config.num_agents, config.action_dim)
        actions = np.clip(actions, -1, 1)
        env_info = config.env.step(actions)[config.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        replay.add(states, actions, rewards, next_states, dones)
        states = next_states
        if np.any(dones):
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

def train_agent(config, agent):
    agent.run_agent()


if __name__=='__main__':
    main()