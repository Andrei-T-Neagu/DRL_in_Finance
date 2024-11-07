import sys
sys.path.insert(0,".")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from neural_networks.FFNN import FFNN
import torch.optim.lr_scheduler as lr_scheduler
import gymnasium as gym
import matplotlib.pyplot as plt

class DDPG_test:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma=0.99, lr=0.0001, batch_size=128, epochs=10, target_update=1, tau=0.1, twin_delayed=False):
        self.state_size = state_size            
        self.action_size = action_size
        self.gamma = gamma                      # discount factor
        self.lr = lr                            # learning rate
        self.batch_size = batch_size            # batch size
        self.epochs = epochs                    # number of epochs
        self.target_update = target_update      # Frequency at which target model is updated
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.twin_delayed = twin_delayed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Policy network
        self.policy = FFNN(state_size, action_size, num_layers, hidden_size).to(self.device)
        self.target_policy = FFNN(state_size, action_size, num_layers, hidden_size).to(self.device)
        
        # Value network
        self.value = FFNN(state_size + 1, 1, num_layers, hidden_size).to(self.device)
        self.target_value = FFNN(state_size + 1, 1, num_layers, hidden_size).to(self.device)

        self.policy.apply(self.init_weights)
        self.value.apply(self.init_weights)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), self.lr)

        self.tau = tau
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())

        if self.twin_delayed:
            self.value2 = FFNN(state_size + 1, 1, num_layers, hidden_size).to(self.device)
            self.value2.apply(self.init_weights)
            self.value2_optimizer = optim.Adam(self.value2.parameters(), self.lr)
            self.target_value2 = FFNN(state_size + 1, 1, num_layers, hidden_size).to(self.device)
            self.target_value2.load_state_dict(self.value2.state_dict())

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    # Store a tuple (s,a,r,s') in the replay memory buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Set the parameters of the target models to be the same as the main models
    def update_target_models(self):
        target_policy_net_state_dict = self.target_policy.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_policy_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_policy_net_state_dict[key]*(1-self.tau)
        self.target_policy.load_state_dict(target_policy_net_state_dict)

        target_value_net_state_dict = self.target_value.state_dict()
        value_net_state_dict = self.value.state_dict()
        for key in value_net_state_dict:
            target_value_net_state_dict[key] = value_net_state_dict[key]*self.tau + target_value_net_state_dict[key]*(1-self.tau)
        self.target_value.load_state_dict(target_value_net_state_dict)

        if self.twin_delayed:
            target_value_net_state_dict = self.target_value2.state_dict()
            value_net_state_dict = self.value2.state_dict()
            for key in value_net_state_dict:
                target_value_net_state_dict[key] = value_net_state_dict[key]*self.tau + target_value_net_state_dict[key]*(1-self.tau)
            self.target_value2.load_state_dict(target_value_net_state_dict)

    def get_action(self, state):
        # Predict mean and log_std
        action = self.policy(state)
        action = torch.tanh(action) * 2.0 + torch.randn(action.shape, device=self.device) * self.epsilon
        action  = torch.clip(action, -2.0, 2.0)
        return action

    # perform an update based on a mini batch sampled from the replay memory buffer
    def replay(self, e):
        # do not perform an update if the replay memory buffer isn't filled enough to sample a batch
        if len(self.memory) < self.batch_size:
            return
        
        # sample a minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.vstack(states).to(self.device)
        actions = torch.vstack(actions).to(self.device)
        rewards = torch.vstack(rewards).to(self.device)
        next_states = torch.vstack(next_states).to(self.device)
        dones = torch.vstack(dones).to(self.device)
    
        with torch.no_grad():
            target_actions = self.target_policy(next_states)
            target_actions = torch.tanh(target_actions) * 2.0
            if self.twin_delayed:
                noise = torch.randn(target_actions.shape, device=self.device) * self.epsilon
                clipped_noise = torch.clamp(noise, -1, 1)
                target_actions = torch.clamp(target_actions + clipped_noise, -2.0, 2.0)
            target_q_values = self.target_value(torch.cat([next_states, target_actions], dim=1))
            if self.twin_delayed:
                target_q_values2 = self.target_value2(torch.cat([next_states, target_actions], dim=1))
                min_target_q_values = torch.min(target_q_values, target_q_values2)
                y = rewards + self.gamma * (1 - dones) * min_target_q_values
            else:
                y = rewards + self.gamma * (1 - dones) * target_q_values
        q_values = self.value(torch.cat([states, actions], dim=1))

        if self.twin_delayed:
            q_values2 = self.value2(torch.cat([states, actions], dim=1))
            value2_loss = nn.HuberLoss()(q_values2, y)

        # current q_value vs expected q_value
        value_loss = nn.HuberLoss()(q_values, y)

        # perform update step on value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        if self.twin_delayed:
            self.value2_optimizer.zero_grad()
            value2_loss.backward()
            self.value2_optimizer.step()

        if self.twin_delayed:
            if e % 2 == 0:
            # policy update
                policy_actions = self.policy(states)
                policy_actions = torch.tanh(policy_actions) * 2.0
                policy_loss = -self.value(torch.cat([states, policy_actions], dim=1)).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
        else:
            # policy update
            policy_actions = self.policy(states)
            policy_actions = torch.tanh(policy_actions) * 2.0
            policy_loss = -self.value(torch.cat([states, policy_actions], dim=1)).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
    def train(self, env, val_env, episodes=500, lr_schedule = True):
        self.policy.train()
        self.value.train()

        episode_losses = []

        if lr_schedule:
            self.value_scheduler = lr_scheduler.LinearLR(self.value_optimizer, start_factor=1.0, end_factor=0.1, total_iters=episodes)
            if self.twin_delayed:
                self.value2_scheduler = lr_scheduler.LinearLR(self.value2_optimizer, start_factor=1.0, end_factor=0.1, total_iters=episodes)
            self.policy_scheduler = lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.1, total_iters=episodes)

        self.epsilon = 0.5
        epsilon_decay = self.epsilon/(episodes+1)

        print("TRAINING DDPG: ")

        for e in range(episodes):
            state, _ = env.reset()
            val_state, _ = val_env.reset()

            state = torch.Tensor([state]).to(self.device)
            val_state = torch.Tensor([val_state]).to(self.device)

            done = torch.zeros(1,1)
            val_done = torch.zeros(1,1)

            total_reward = torch.zeros(1, 1, device=self.device)
            val_total_reward = torch.zeros(1, 1, device=self.device)
            
            while torch.all(done == 0):

                with torch.no_grad():
                    action = self.get_action(state)
                    val_action = self.policy(val_state)
                    val_action = torch.tanh(val_action) * 2.0

                next_state, reward, terminated, truncated, _ = env.step([action.item()])
                val_next_state, val_reward, terminated, truncated, _ = val_env.step([val_action.item()])

                next_state = torch.Tensor([next_state]).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                done = terminated or truncated
                done = torch.Tensor([done]).to(self.device)
                
                self.remember(state, action, reward, next_state, done)

                val_next_state = torch.Tensor([val_next_state]).to(self.device)
                val_reward = torch.Tensor([val_reward]).to(self.device)

                state = next_state
                val_state = val_next_state

                total_reward += reward
                val_total_reward += val_reward

                self.replay(e)

            episode_losses.append(total_reward.item())

            if e % self.target_update == 0:
                self.update_target_models()
            
            if lr_schedule and len(self.memory) > self.batch_size:
                self.value_scheduler.step()
                if self.twin_delayed:
                    self.value2_scheduler.step()
                self.policy_scheduler.step()
            
            if self.epsilon > 0.01:
                self.epsilon -= epsilon_decay*2

            if e % 10 == 0:
                print(f"Episode {e}/{episodes-1}, Epsilon: {self.epsilon}, Total Reward: {total_reward.item()}")
        
        return episode_losses

    def test(self, env, episodes=1000, render=False):
        """
        Test the trained PPO agent in the environment.
        
        Args:
            env: The environment to test on.
            episodes: Number of episodes to run for testing.
            render: Whether to render the environment.
            
        Returns:
            avg_reward: Average reward per episode during testing.
        """
        self.policy.eval()

        total_val_reward = torch.zeros(1, device=self.device)

        print("TESTING DDPG: ")
        for e in range(episodes):
            state, _ = env.reset()
            state = torch.Tensor([state]).to(self.device)
            done = torch.zeros(1)
            total_reward = torch.zeros(1, device=self.device)
            
            i = 0
            while torch.all(done == 0):

                # Get the action from the trained model
                with torch.no_grad():
                    action = self.policy(state)
                    action = torch.tanh(action) * 2.0

                # Step the environment with the chosen action
                next_state, reward, terminated, truncated, _ = env.step([action.item()])
                
                next_state = torch.Tensor([next_state]).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                done = terminated or truncated
                done = torch.Tensor([done]).to(self.device)

                state = next_state

                total_reward += reward

            print(f"Episode {e}/{episodes-1}, Total Reward: {total_reward.item()}")

    def save(self, name):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'value_state_dict': self.value.state_dict()}, name)

    def load(self, name):
        checkpoint = torch.load(name)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])

num_layers = 3
nbs_units = 128
lr = 0.001
batch_size = 128
twin_delayed = True

#for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

env = gym.make("Pendulum-v1")
env.reset(seed=0)
val_env = gym.make("Pendulum-v1")
val_env.reset(seed=1)

ddpg_agent = DDPG_test(state_size=3, action_size=1, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size, twin_delayed=twin_delayed)
ddpg_train_losses = ddpg_agent.train(env, val_env, episodes=1000)
ddpg_agent.test(env, episodes=100)

ma_size = 100
hyperparameter_path = "/home/a_eagu/DRL_in_Finance/ddpg_hyperparameters/"
ddpg_losses = np.convolve(ddpg_train_losses, np.ones(ma_size), 'valid') / ma_size
ddpg_train_losses_fig = plt.figure(figsize=(12, 6))
plt.plot(ddpg_train_losses, label="Total Reward")
plt.plot(ddpg_losses, label="Total Reward (Moving Average)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.title("Total Reward " + str(ma_size) + " Episode Moving Average for DDPG")
plt.savefig(hyperparameter_path + "training_losses/ddpg_PENDULUM_train_losses.png")
plt.close()