import numpy as np
import random
import copy
from collections import namedtuple, deque
from module_model_tennis import ModuleActor, ModuleCritic
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
LEARN_EVERY = 1
LEARN_NB = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ObjectAgent(object):
    def __init__(self, data_state_size, data_action_size, data_random_seed):
        self.data_state_size = data_state_size
        self.data_action_size = data_action_size
        self.data_seed = random.seed(data_random_seed)
        self.data_i_learn = 0
        self.data_actor_local = ModuleActor(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_actor_target = ModuleActor(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_actor_optimizer = optim.Adam(self.data_actor_local.parameters(), lr=LR_ACTOR)
        self.data_critic_local = ModuleCritic(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_critic_target = ModuleCritic(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_critic_optimizer = optim.Adam(self.data_critic_local.parameters(), lr=LR_CRITIC)
        self.copy_weights(self.data_critic_local, self.data_critic_target)
        self.copy_weights(self.data_actor_local, self.data_actor_target)
        self.data_noise = ObjectOUNoise(2 * data_action_size, data_random_seed)
        self.data_memory = ObjectReplayBuffer(BUFFER_SIZE, BATCH_SIZE, data_random_seed)

    def learn(self, data_experiences, data_gamma):
        data_states, data_actions, data_actions_other_player, data_rewards, data_next_states, data_next_states_other_player, data_dones = data_experiences
        data_actions_next = self.data_actor_target(data_next_states)
        data_actions_next_other_player = self.data_actor_target(data_next_states_other_player)
        data_Q_targets_next = self.data_critic_target(data_next_states, data_actions_next, data_actions_next_other_player)
        data_Q_targets = data_rewards + (data_gamma * data_Q_targets_next * (1 - data_dones))
        data_Q_expected = self.data_critic_local(data_states, data_actions, data_actions_other_player)
        data_critic_loss = F.mse_loss(data_Q_expected, data_Q_targets)
        self.data_critic_optimizer.zero_grad()
        data_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.data_critic_local.parameters(), 1)
        self.data_critic_optimizer.step()
        data_actions_pred = self.data_actor_local(data_states)
        data_actor_loss = -self.data_critic_local(data_states, data_actions_pred, data_actions_other_player).mean()
        self.data_actor_optimizer.zero_grad()
        data_actor_loss.backward()
        self.data_actor_optimizer.step()
        self.soft_update(self.data_critic_local, self.data_critic_target, TAU)
        self.soft_update(self.data_actor_local, self.data_actor_target, TAU)

    def soft_update(self, data_local_model, data_target_model, tau):
        for i_target_param, i_local_param in zip(data_target_model.parameters(), data_local_model.parameters()):
            i_target_param.data.copy_(tau * i_local_param.data + (1.0 - tau) * i_target_param.data)

    def copy_weights(self, data_source, data_target):
        for i_target_param, i_source_param in zip(data_target.parameters(), data_source.parameters()):
            i_target_param.data.copy_(i_source_param.data)

    def act(self, data_state, data_add_noise=True, data_noise_factor=1.0):
        data_state = torch.from_numpy(data_state).float().to(device)
        self.data_actor_local.eval()
        with torch.no_grad():
            data_action = self.data_actor_local(data_state).cpu().data.numpy()
        self.data_actor_local.train()
        if data_add_noise:
            data_action += data_noise_factor * self.data_noise.sample().reshape((-1, 2))
        return np.clip(data_action, -1, 1)

    def reset(self):
        self.data_noise.reset()

    def step(self, data_states, data_actions, data_actions_other_player, data_rewards, data_next_states, data_next_states_other_players, data_dones):
        for i_state, i_action, i_action_other_player, i_reward, i_next_state, i_next_state_other_player, i_done \
                in zip(data_states, data_actions, data_actions_other_player, data_rewards, data_next_states, data_next_states_other_players, data_dones):
            self.data_memory.add(i_state, i_action, i_action_other_player, i_reward, i_next_state, i_next_state_other_player, i_done)
        self.data_i_learn = (self.data_i_learn + 1) % LEARN_EVERY
        if len(self.data_memory) > BATCH_SIZE and self.data_i_learn == 0:
            for _ in range(LEARN_NB):
                data_experiences = self.data_memory.sample()
                self.learn(data_experiences, GAMMA)


class ObjectOUNoise:
    def __init__(self, data_size, data_seed, data_mu=0., data_theta=0.15, data_sigma=0.2):
        self.data_mu = data_mu * np.ones(data_size)
        self.data_theta = data_theta
        self.data_sigma = data_sigma
        self.data_seed = random.seed(data_seed)
        self.data_state = None
        self.reset()

    def sample(self):
        data_x = self.data_state
        data_dx = self.data_theta * (self.data_mu - data_x) + self.data_sigma * np.array([random.random() for i in range(len(data_x))])
        self.data_state = data_x + data_dx
        return self.data_state

    def reset(self):
        self.data_state = copy.copy(self.data_mu)


class ObjectReplayBuffer:
    def __init__(self, data_buffer_size, data_batch_size, data_seed):
        self.data_memory = deque(maxlen=data_buffer_size)
        self.data_batch_size = data_batch_size
        self.data_experience = namedtuple("Experience", field_names=["state", "action", "action_other_player", "reward", "next_state", "next_state_other_player", "done"])
        self.data_seed = random.seed(data_seed)

    def sample(self):
        data_experiences = random.sample(self.data_memory, k=self.data_batch_size)
        data_states = torch.from_numpy(np.vstack([e.state for e in data_experiences if e is not None])).float().to(device)
        data_actions = torch.from_numpy(np.vstack([e.action for e in data_experiences if e is not None])).float().to(device)
        data_actions_other_player = torch.from_numpy(np.vstack([e.action_other_player for e in data_experiences if e is not None])).float().to(device)
        data_rewards = torch.from_numpy(np.vstack([e.reward for e in data_experiences if e is not None])).float().to(device)
        data_next_states = torch.from_numpy(np.vstack([e.next_state for e in data_experiences if e is not None])).float().to(
            device)
        data_next_states_other_player = torch.from_numpy(np.vstack([e.next_state_other_player for e in data_experiences if e is not None])).float().to(
            device)
        data_dones = torch.from_numpy(np.vstack([e.done for e in data_experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return data_states, data_actions, data_actions_other_player, data_rewards, data_next_states, data_next_states_other_player, data_dones

    def add(self, data_state, data_action, data_action_other_player, data_reward, data_next_state, data_next_state_other_player, data_done):
        data_e = self.data_experience(data_state, data_action, data_action_other_player, data_reward, data_next_state, data_next_state_other_player, data_done)
        self.data_memory.append(data_e)

    def __len__(self):
        return len(self.data_memory)
