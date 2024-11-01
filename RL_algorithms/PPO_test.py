import sys
sys.path.insert(0,".")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from neural_networks.FFNN import FFNN
import torch.optim.lr_scheduler as lr_scheduler
from neural_networks.CategoricalActor import CategoricalFFNN
import gymnasium as gym

class PPO_test:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma=1.0, lr=0.0001, clip_eps=0.2, batch_size=128, epochs=10, lambd=0.95):
        self.state_size = state_size            
        self.action_size = action_size
        self.gamma = gamma                      # discount factor               
        self.lambd = lambd                      # lambda used in the Generalized Advantage Estimate  
        self.clip_eps = clip_eps                # clipping factor of the gradient estimate
        self.lr = lr                            # learning rate
        self.batch_size = batch_size            # batch size
        self.epochs = epochs                    # number of epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Policy network
        self.policy = CategoricalFFNN(state_size, action_size, num_layers, hidden_size).to(self.device)
        # Value network
        self.value = FFNN(state_size, 1, num_layers, hidden_size).to(self.device)

        self.policy.apply(self.init_weights)
        self.value.apply(self.init_weights)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), self.lr)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    def get_action(self, state):
        # Sample from distribution
        dist = self.policy(state)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, action_log_prob, entropy

    def calculate_returns(self, rewards, normalize = True):
        
        episode_length = rewards.shape[1]
        returns = torch.zeros(self.batch_size, episode_length, device=self.device)
        
        for i in reversed(range(episode_length)):
            if i == episode_length-1:
                returns[:, i] = rewards[:, i]
            else:
                returns[:, i] = rewards[:, i] + returns[:, i+1] * self.gamma

        if normalize:
            returns = (returns - returns.mean()) / returns.std()
            
        return returns

    def calculate_advantages(self, returns, values, normalize = True):
        
        advantages = returns - values
        
        if normalize:
            
            advantages = (advantages - advantages.mean()) / advantages.std()
            
        return advantages

    def train(self, env, val_env, episodes=1000, lr_schedule=True):
        self.policy.train()
        self.value.train()
        
        episode_val_loss = []

        if lr_schedule:
            self.value_scheduler = lr_scheduler.LinearLR(self.value_optimizer, start_factor=1.0, end_factor=0.1, total_iters=episodes)
            self.policy_scheduler = lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.1, total_iters=episodes)
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = torch.Tensor([state]).to(self.device)
            val_state, _ = val_env.reset()
            val_state = torch.Tensor([val_state]).to(self.device)

            val_total_reward = torch.zeros(self.batch_size, device=self.device)

            done = torch.zeros(self.batch_size, device=self.device)
            val_done = torch.zeros(self.batch_size, device=self.device)

            # Rollout
            step = 0
            while torch.all(val_done == 0):
                with torch.no_grad():
                    dist = self.policy(val_state)
                    val_action = dist.sample()
                
                val_next_state, val_reward, terminated, truncated, _ = val_env.step(val_action.item())
                val_next_state = torch.Tensor([val_next_state]).to(self.device)
                val_reward = torch.Tensor([val_reward]).to(self.device)
                val_state = val_next_state
                val_total_reward += val_reward
                val_done = terminated or truncated
                val_done = torch.Tensor([val_done]).to(self.device)
            
            while torch.all(done == 0):
                with torch.no_grad():
                    action, action_log_prob, _ = self.get_action(state)
                    value = self.value(state)

                    next_state, reward, terminated, truncated, _ = env.step(action.item())

                    next_state = torch.Tensor([next_state]).to(self.device)
                    reward = torch.Tensor([reward]).to(self.device)
                    done = terminated or truncated
                    done = torch.Tensor([done]).to(self.device)

                    if step == 0:
                        rewards = reward.unsqueeze(1)
                        actions = action.unsqueeze(1)
                        states = state.unsqueeze(1)
                        log_probs = action_log_prob.unsqueeze(1)
                        dones = done.unsqueeze(1)
                        values = value
                    else:
                        rewards = torch.cat((rewards, reward.unsqueeze(1)), 1)
                        actions = torch.cat((actions, action.unsqueeze(1)), 1)
                        states = torch.cat((states, state.unsqueeze(1)), 1)
                        log_probs = torch.cat((log_probs, action_log_prob.unsqueeze(1)), 1)
                        dones = torch.cat((dones, done.unsqueeze(1)), 1)
                        values = torch.cat((values, value), 1)
                    
                    state = next_state
                    step += 1
            
            val_loss = val_total_reward

            episode_val_loss.append(val_loss.item())

            # Compute advantages and returns
            returns = self.calculate_returns(rewards)
            advantages = self.calculate_advantages(returns, values)
            
            rewards = rewards.flatten(start_dim=0, end_dim=1)
            actions = actions.flatten(start_dim=0, end_dim=1)
            states = states.flatten(start_dim=0, end_dim=1)
            log_probs = log_probs.flatten(start_dim=0, end_dim=1)
            dones = dones.flatten(start_dim=0, end_dim=1)
            values = values.flatten(start_dim=0, end_dim=1)
            returns = returns.flatten(start_dim=0, end_dim=1)
            advantages = advantages.flatten(start_dim=0, end_dim=1)

            # print("rewards.shape: ", rewards.shape)
            # print("actions.shape: ", actions.shape) 
            # print("states.shape: ", states.shape)
            # print("log_probs.shape: ", log_probs.shape)
            # print("dones.shape: ", dones.shape)
            # print("values.shape: ", values.shape)
            # print("returns.shape: ", returns.shape)
            # print("advantages.shape: ", advantages.shape)

            # Training using mini-batches and multiple epochs
            for _ in range(self.epochs):

                # Get new log probabilities
                
                dist = self.policy(states)
                new_log_probs = dist.log_prob(actions)
                dist_entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - log_probs)
                # Clip the ratio
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                loss_policy = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                # Value loss
                value_pred = self.value(states).flatten()
                loss_value = nn.HuberLoss()(value_pred, returns)

                # # Combine losses and calculate the surrogate loss
                surrogate_loss = loss_policy + 0.01 * loss_value + 0.01 * dist_entropy

                # Update the networks
                self.policy_optimizer.zero_grad()
                loss_policy.backward()
                self.policy_optimizer.step()
                self.value_optimizer.zero_grad()
                loss_value.backward()
                self.value_optimizer.step()

            if lr_schedule:
                self.value_scheduler.step()
                self.policy_scheduler.step()

            if episode % 1 == 0:
                print(f"Episode {episode}/{episodes}, Policy Loss: {loss_policy.item()}, Value Loss: {loss_value.item()}, Validation Loss: {val_loss.item()}")

        return episode_val_loss

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

        print("TESTING PPO: ")
        for episode in range(episodes):
            state, _ = env.reset()
            
            state = torch.Tensor([state]).to(self.device)
            done = torch.zeros(self.batch_size, device=self.device)
            
            total_reward = torch.zeros(self.batch_size, device=self.device)
            
            i = 0
            while torch.all(done == 0):

                # Get the action from the trained model
                with torch.no_grad():
                    dist = self.policy(state)
                    action = dist.sample()

                # Step the environment with the chosen action
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                
                next_state = torch.Tensor([next_state]).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                done = terminated or truncated
                done = torch.Tensor([done]).to(self.device)
                
                # Accumulate the reward
                total_reward += reward
                # Move to the next state
                state = next_state
                
                i += 1
            
            loss = torch.sqrt(torch.mean(torch.square(torch.where(total_reward > 0, total_reward, 0))))

            if episode % 10 == 0:
                print(f"Episode: {episode}/{episodes-1}, Total Reward: {loss.item()}")
    

    def save(self, name):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'value_state_dict': self.value.state_dict()}, name)

    def load(self, name):
        checkpoint = torch.load(name)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])

num_layers = 2
nbs_units = 128
lr = 0.001
batch_size = 1

#for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

env = gym.make("CartPole-v1")
env.reset(seed=0)
val_env = gym.make("CartPole-v1")
val_env.reset(seed=1)

dqn_agent = PPO_test(state_size=4, action_size=2, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
dqn_agent.train(env, val_env, episodes=500)
dqn_agent.test(env, episodes=100)