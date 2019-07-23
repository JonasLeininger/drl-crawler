import torch
import torch.nn.functional as F
import numpy as np
import random

from models.ddpg_actor import DDPGActor
from models.ddpg_critic import DDPGCritic
from replay_buffer import ReplayBuffer
from ornstein_uhlenbeck_noise import Noise

class DDPGAgent():

    def __init__(self, config):
        self.config = config
        self.checkpoint_path_actor = "checkpoints/ddpg/cp-actor-{epoch:04d}.pt"
        self.checkpoint_path_critic = "checkpoints/ddpg/cp-critic-{epoch:04d}.pt"
        self.weights_path_actor = "weights/ddpg/cp-actor-{epoch:04d}.pt"
        self.weights_path_critic = "weights/ddpg/cp-critic-{epoch:04d}.pt"
        self.episodes = 5000
        self.env_info = None
        self.env_agents = None
        self.states = None
        self.dones = None
        self.loss = None
        self.gamma = 0.99
        self.tau = 0.01
        self.batch_size = self.config.config['BatchesSizeDDPG']
        self.memory = ReplayBuffer(100000, self.batch_size)
        self.learn_every = 5
        self.num_learn = 20
        self.actor_local = DDPGActor(config)
        self.actor_target = DDPGActor(config)
        self.critic_local = DDPGCritic(config)
        self.critic_target = DDPGCritic(config)
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=float(self.config.config['LearningRateDDPG']),
                                                weight_decay=0.0)
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=float(self.config.config['LearningRateDDPG']),
                                                 weight_decay=0.0000)

        self.hard_update(self.critic_local, self.critic_target)
        self.hard_update(self.actor_local, self.actor_target)
        self.seed = random.seed(16)
        self.noise = Noise(20, self.seed)
        self.scores = []
        self.scores_agent_mean = []

    def run_agent(self):
        for step in range(self.episodes):
            print("Episonde {}/{}".format(step, self.episodes))
            self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
            self.env_agents = self.env_info.agents
            self.states = self.env_info.vector_observations
            self.dones = self.env_info.local_done
            self.run_training()
            print("Average score from 20 agents: >> {:.2f} <<".format(self.scores_agent_mean[-1]))
            if (step+1)%100==0:
                self.save_checkpoint(step+1)
                np.save(file="checkpoints/ddpg/ddpg_save_dump.npy", arr=np.asarray(self.scores))

            if (step + 1) >= 100:
                self.mean_of_mean = np.mean(self.scores_agent_mean[-100:])
                print("Mean of the last 100 episodes: {:.2f}".format(self.mean_of_mean))
                if self.mean_of_mean>=400.0:
                    print("Solved the environment after {} episodes with a mean of {:.2f}".format(step, self.mean_of_mean))
                    np.save(file="checkpoints/ddpg/ddpg_final.npy", arr=np.asarray(self.scores))
                    self.save_checkpoint(step+1)
                    break

    def run_training(self):
        t_step = 0
        scores = np.zeros((1, 12))
        self.noise.reset()
        while not np.any(self.dones):
        # for t in range(1000):
            action_prediction = self.act(self.states)
            # print(action_prediction)
            self.env_info = self.config.env.step(action_prediction)[self.config.brain_name]
            next_states = self.env_info.vector_observations
            self.dones = self.env_info.local_done
            # dones_binary = np.vstack([done for done in self.dones if done is not None])
            rewards = self.env_info.rewards
            self.memory.add(self.states, action_prediction, rewards, next_states, self.dones)
            scores = scores + rewards

            if t_step%self.learn_every==0:
                for _ in range(self.num_learn):
                    self.learn()
            self.states = next_states
            t_step += 1
        print("Number of steps {}".format(t_step))
        print("Agent scores:")
        print(scores)
        self.scores.append(scores)
        self.scores_agent_mean.append(scores.mean())

    def act(self, states, add_noise=True):
        states = torch.tensor(states, dtype=torch.float, device=self.actor_local.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            action += noise
        return np.clip(action, -1, 1)

    def learn(self):
        if len(self.memory.memory) > self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample()
            target_actions = self.actor_target(next_states)
            q_targets_next = self.critic_target(next_states, target_actions)
            q_targets = rewards + (self.gamma * q_targets_next * (1- dones))
            q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(q_expected, q_targets)

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.optimizer_critic.step()

            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_checkpoint(self, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.actor_target.state_dict(),
            'optimizer_state_dict': self.optimizer_actor.state_dict()
        }, self.checkpoint_path_actor.format(epoch=epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.critic_target.state_dict(),
            'optimizer_state_dict': self.optimizer_critic.state_dict()
        }, self.checkpoint_path_critic.format(epoch=epoch))

    def load_checkpoint(self, saved_episode: str):
        checkpoint = torch.load(self.weights_path_actor.format(epoch=saved_episode))
        self.actor_target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_state_dict'])

        self.actor_target.eval()

        checkpoint = torch.load(self.weights_path_actor.format(epoch=saved_episode))
        self.actor_local.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_state_dict'])

        self.actor_local.eval()

        checkpoint = torch.load(self.weights_path_critic.format(epoch=saved_episode))
        self.critic_target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_state_dict'])

        self.critic_target.eval()

        checkpoint = torch.load(self.weights_path_critic.format(epoch=saved_episode))
        self.critic_local.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_state_dict'])

        self.critic_local.eval()
