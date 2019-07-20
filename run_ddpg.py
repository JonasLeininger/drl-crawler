import numpy as np

from config.config import Config


def main():
    config = Config()
    print_env_information(config)
    print(config.state_dim)
    run_random_env(config)
    config.env.close()


def print_env_information(config):
    config.env_info = config.env.reset(train_mode=False)[config.brain_name]
    config.num_agents = len(config.env_info.agents)
    print('Number of agents:', config.num_agents)
    print('Size of each action:', config.action_dim)
    config.states = config.env_info.vector_observations
    print('There are {} agents. Each observes a state with length: {}'.format(config.states.shape[0], config.state_dim))
    print('The state for the first agent looks like:', config.states[0])


def run_random_env(config):
    # env_info = config.env.reset(train_mode=False)[config.brain_name]
    # states = env_info.vector_observations
    scores = np.zeros(config.num_agents)
    steps = 0
    # for t in range(steps):
    while True:
        steps += 1
        actions = np.random.randn(config.num_agents, config.action_dim)
        actions = np.clip(actions, -1, 1)
        env_info = config.env.step(actions)[config.brain_name]
        # next_states = env_info.vector_observations
        # rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        # replay.add(states, actions, rewards, next_states, dones)
        # states = next_states
        if np.any(dones):
            print('done after {} steps'.format(steps))
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


if __name__ == '__main__':
    main()
