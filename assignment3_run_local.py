import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


# Hyper parameters that will be used in the DQN algorithm
LEARNING_RATE = 0.001
GAMMA = 0.99  # discount factor
MEM_SIZE = 100000
BATCH_SIZE = 64
FC1_DIMS = 128  # First fully connected layer dimensions
FC2_DIMS = 128  # Second fully connected layer dimensions
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 200
REPLAY_START_SIZE = 10000
NETWORK_UPDATE_ITERS = 1000
MEM_RETAIN = 0.1  # Retain the first 10% of the replay buffer to avoid catastrophic forgetting


FC1_DIMS = 128                   # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 128                   # Number of neurons in our MLP's second hidden layer

# metrics for displaying training status
best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
np.bool = np.bool_

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# for creating the policy and target networks - same architecture
class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        # build an MLP with 2 hidden layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),   # input layer
            torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),    # hidden layer
            torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC2_DIMS, self.action_space)    # output layer
            )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # loss function

    def forward(self, x):
        return self.layers(x)

# handles the storing and retrival of sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            mem_index = int(self.mem_count % ((1-MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))  # avoid catastrophic forgetting, retain first 10% of replay buffer

        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())  # initially set weights of Q to \hat{Q}
        self.learn_count = 0    # keep track of the number of iterations we have learnt for

    # epsilon greedy
    def choose_action(self, observation):
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0
        
        if random.random() < eps_threshold:
            # Exploration: Randomly choosing an action
            return env.action_space.sample()  # This is a more generic approach
        else:
            # Exploitation: Choose the best known action
            state = torch.tensor([observation], dtype=torch.float32)  # Minor adjustment for tensor shape
            self.policy_network.eval()
            with torch.no_grad():
                q_values = self.policy_network(state)
            return torch.argmax(q_values).item()
    

    # main training loop
    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()  # retrieve random batch of samples from replay memory
        states = torch.tensor(states , dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(states)                # get current q-value estimates (all actions) from policy network, Q
        q_values = q_values[batch_indices, actions]           # q values for sampled actions only

        self.target_network.eval()                            # only need forward pass
        with torch.no_grad():                                 # so we don't compute gradients - save memory and computation
            q_values_next = self.target_network(states_)      # target q-values for states_ for all actions (target network, \hat{Q})

        q_values_next_max = torch.max(q_values_next, dim=1)[0]  # max q values for next state

        q_target = rewards + GAMMA * q_values_next_max * dones  # our target q-value

        loss = self.policy_network.loss(q_target, q_values)     # compute loss between target (target network, \hat{Q}) and estimated q-values (policy network, Q)
        #compute gradients and update policy network Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        # set target network \hat{Q}'s weights to policy network Q's weights every C steps
        if  self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            print("updating target network")
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate



def preprocess_state(state):
    # Example preprocessing: Flatten the state if it's a nested structure
    # Adjust this example to match your specific observation structure
    flattened_state = np.hstack(state)
    return np.array(flattened_state, dtype=np.float32)

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
##########################################################################################################################

# Local en 
# gym.register(id='SimpleDrivingEnv',entry_point='simple_car_a3.simple_driving.envs.simp1e_driving_env:simp1eDrivingEnv') 
# # ith GU 
# env = gym.make('SimpleDrivingEnv', apply_api_compatibi1ity=True, renders=True, isDiscrete=True) 
# ithout GU t en = gym.make(' simpleDrivingEnv')

# state, info = env.reset()

# state = env.reset()
# try:
#     state_array = np.array(state, dtype=np.float32)
# except ValueError as e:
#     print("Error converting state to numpy array:", e)
#     # Implement a fallback or preprocessing step here
#     # For example, if state is a list of lists with inconsistent lengths, you might need to preprocess it
#     # This is just a placeholder and likely needs to be customized
#     state_array = preprocess_state(state)  # Define preprocess_state according to your needs


env.action_space.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
episode_batch_score = 0
episode_reward = 0
agent = DQN_Solver(env)

num_episodes = 1000  # Define the number of episodes for training

for episode in range(num_episodes):
    state = env.reset()
    # Ensure state is a numpy array
    state = np.array(state, dtype=np.float32)
    total_reward = 0

    for t in range(200):  # Limit the number of steps per episode
        action = agent.choose_action(state)
        next_state, reward, done, _, info = env.step(action)
        
        # Convert next_state to numpy array
        next_state = np.array(next_state, dtype=np.float32)

        agent.memory.add(state, action, reward, next_state, done)
        
        if agent.memory.mem_count > REPLAY_START_SIZE:
            agent.learn()

        state = next_state  # next_state is already a numpy array here
        total_reward += reward

        if done:
            break

    print(f"Episode {episode+1}, Total reward: {total_reward}, Epsilon: {agent.returning_epsilon()}")

    # Update the target network
    if episode % NETWORK_UPDATE_ITERS == 0:
        agent.update_target_network()

env.close()





