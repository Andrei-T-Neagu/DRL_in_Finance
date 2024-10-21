import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neural_networks.FFNN import FFNN
from option_hedging.code_pytorch.DeepHedgingEnvironment import DeepHedgingEnvironment

class PPO:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma=0.99, lr=0.0003, clip_eps=0.2, batch_size=512, epochs=10, lambd=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lambd = lambd
        self.clip_eps = clip_eps
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Policy network
        self.policy = FFNN(state_size, action_size * 2, num_layers, hidden_size).to(self.device) # Output mean and log_std
        # Value network
        self.value = FFNN(state_size, 1, num_layers, hidden_size).to(self.device)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)

    def get_action(self, state):
        # Predict mean and log_std
        mean_log_std = self.policy(state)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        std = torch.exp(log_std)
        # Sample from Gaussian distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, action_log_prob

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.gamma * (1 - dones[t]) * values[t + 1] - values[t]
            advantage = td_error + self.gamma * self.lambd * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        return advantages

    def train(self, env, episodes=1000):
        for episode in range(episodes):
            state = env.reset(self.batch_size)
            done = torch.zeros(self.batch_size, device=self.device)
            rewards, actions, states, log_probs, dones, values = [], [], [], [], [], []

            # Rollout
            while torch.all(done == 0):
                state_tensor = torch.FloatTensor(state).to(self.device)
                action, action_log_prob = self.get_action(state_tensor)
                value = self.value(state_tensor)

                next_state, reward, done = env.step(action)

                rewards.append(reward)
                actions.append(action)
                states.append(state_tensor)
                log_probs.append(action_log_prob)
                dones.append(done)
                values.append(value)

                state = next_state

            values.append(self.value(torch.FloatTensor(next_state).to(self.device))) # For GAE

            # Compute advantages and returns
            advantages = self.compute_advantages(rewards, values, dones)
            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            # Training using mini-batches and multiple epochs
            for _ in range(self.epochs):
                for idx in range(0, len(states), self.batch_size):
                    sampled_states = torch.cat(states[idx:idx + self.batch_size])
                    sampled_actions = torch.cat(actions[idx:idx + self.batch_size])
                    sampled_log_probs = torch.cat(log_probs[idx:idx + self.batch_size])
                    sampled_returns = torch.cat(returns[idx:idx + self.batch_size])
                    sampled_advantages = torch.cat(advantages[idx:idx + self.batch_size])

                    # Get new log probabilities
                    new_actions, new_log_probs = self.get_action(sampled_states)
                    ratio = torch.exp(new_log_probs - sampled_log_probs)

                    # Clip the ratio
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    loss_policy = -torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages).mean()

                    # Value loss
                    value_pred = self.value(sampled_states).squeeze()
                    loss_value = nn.MSELoss()(value_pred, sampled_returns)

                    # Update the networks
                    self.optimizer_policy.zero_grad()
                    loss_policy.backward()
                    self.optimizer_policy.step()

                    self.optimizer_value.zero_grad()
                    loss_value.backward()
                    self.optimizer_value.step()

            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes}, Policy Loss: {loss_policy.item()}, Value Loss: {loss_value.item()}")

        self.save("ppo_model.pth")

    def save(self, name):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'value_state_dict': self.value.state_dict()}, name)

    def load(self, name):
        checkpoint = torch.load(name)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])