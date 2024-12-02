import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neural_networks.FFNN import FFNN
from option_hedging.DeepHedgingEnvironment import DeepHedgingEnvironment
import torch.optim.lr_scheduler as lr_scheduler

class PPO:
    def __init__(self, config, state_size=3, action_size=1, clip_eps=0.2, epochs=10, device='cpu'):
        
        self.lr = config.get("lr", 0.0001)                          # learning rate
        self.batch_size = config.get("batch_size", 128)             # batch size
        self.num_layers = config.get("num_layers")
        self.hidden_size = config.get("hidden_size")
        
        self.gamma = 1.0                                            # discount factor
        self.state_size = state_size            
        self.action_size = action_size                 
        self.clip_eps = clip_eps                # clipping factor of the gradient estimate
        self.epochs = epochs                    # number of epochs

        self.device = device
        # Policy network
        self.policy = FFNN(state_size, action_size * 2, self.num_layers, self.hidden_size, log_predicted=True).to(self.device) # Output mean and log_std
        # Value network
        self.value = FFNN(state_size, 1, self.num_layers, self.hidden_size).to(self.device)

        self.policy.apply(self.init_weights)
        self.value.apply(self.init_weights)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), self.lr)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    def get_action(self, state):
        # Predict mean and log_std
        mean_log_std = self.policy(state)
        # print("mean_log_std: ", mean_log_std)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        std = torch.exp(log_std)
        # Sample from Gaussian distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        # print("mean: ", mean)
        # print("std: ", std)
        # print("action_log_prob: ", action_log_prob)
        entropy = dist.entropy().mean()
        return action, action_log_prob, entropy

    def calculate_returns(self, rewards, normalize = False):
    
        returns = torch.zeros(self.batch_size, self.N, device=self.device)
        
        for i in reversed(range(self.N)):
            if i == self.N-1:
                returns[:, i] = rewards[:, i]
            else:
                returns[:, i] = rewards[:, i] + returns[:, i+1] * self.gamma

        if normalize:
            returns = (returns - returns.mean()) / returns.std()
            
        return returns

    def calculate_advantages(self, returns, values, normalize = False):
        
        advantages = returns - values
        
        if normalize:
            
            advantages = (advantages - advantages.mean()) / advantages.std()
            
        return advantages

    def train(self, env, val_env, BS_rsmse, episodes=1000, lr_schedule=True, render=False):
        self.policy.train()
        self.value.train()
        env.train()
        val_env.train()

        self.N = env.N
        
        episode_val_loss = []

        if lr_schedule:
            self.value_scheduler = lr_scheduler.LinearLR(self.value_optimizer, start_factor=1.0, end_factor=0.005, total_iters=episodes)
            self.policy_scheduler = lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.005, total_iters=episodes)
        
        print("TRAINING PPO")

        for episode in range(episodes):
            state = env.reset(self.batch_size)

            done = torch.zeros(self.batch_size, device=self.device)
            rewards = torch.zeros(self.batch_size, self.N, device=self.device)
            actions = torch.zeros(self.batch_size, self.N, self.action_size, device=self.device)
            states = torch.zeros(self.batch_size, self.N, self.state_size, device=self.device)
            log_probs = torch.zeros(self.batch_size, self.N, 1, device=self.device)
            dones = torch.zeros(self.batch_size, self.N, device=self.device)
            values = torch.zeros(self.batch_size, self.N, device=self.device)

            # Rollout
            step = 0
            while torch.all(done == 0):
                with torch.no_grad():
                    action, action_log_prob, _ = self.get_action(state)
                    value = self.value(state).flatten()

                    next_state, reward, done = env.step(action)
                    rewards[:, step] = -torch.square(torch.where(reward > 0, reward, -0))
                    actions[:, step, :] = action
                    states[:, step, :] = state
                    log_probs[:, step, :] = action_log_prob
                    dones[:, step] = done
                    values[:, step] = value

                    state = next_state
                    step += 1

            # Compute advantages and returns
            returns = self.calculate_returns(rewards).to(self.device)
            advantages = self.calculate_advantages(returns, values)

            # Training using mini-batches and multiple epochs
            for _ in range(self.epochs):

                # Get new log probabilities
                
                mean_log_std = self.policy(states)
                mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions)
                dist_entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - log_probs).squeeze()

                # Clip the ratio
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                loss_policy = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                # Value loss
                value_pred = self.value(states).squeeze(-1)
                loss_value = nn.MSELoss()(value_pred, returns)

                # # Combine losses and calculate the surrogate loss
                # surrogate_loss = loss_policy + 0.5 * loss_value + 0.01 * dist_entropy

                # Update the networks
                self.value_optimizer.zero_grad()
                loss_value.backward()
                self.value_optimizer.step()
                self.policy_optimizer.zero_grad()
                loss_policy.backward()
                self.policy_optimizer.step()
            
            if lr_schedule:
                self.value_scheduler.step()
                self.policy_scheduler.step()

            # Compute validation losses
            if episode % 1000 == 0:
                _, _, val_rsmse = self.test(val_env)
                self.policy.train()
                episode_val_loss.append(val_rsmse)

            if render and episode % 10000 == 0:
                print(f"Episode {episode}/{episodes-1}, Policy Loss: {loss_policy.item()}, Value Loss: {loss_value.item()}, Validation Loss: {val_rsmse}")

            # Early stopping
            if len(episode_val_loss) > 10 and val_rsmse < BS_rsmse:
                if min(episode_val_loss[:-10]) < min(episode_val_loss[-10:]):
                    break

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
        
        env.test()
        train_size = env.dataset.shape[0]
        num_points = env.N
        batches = int(train_size/self.batch_size)

        actions = torch.zeros(num_points, self.batch_size, batches, device=self.device)
        rewards = torch.zeros(self.batch_size, batches, device=self.device)

        total_val_reward = torch.zeros(self.batch_size, batches, device=self.device)

        for batch in range(batches):
            state = env.reset(self.batch_size)
            done = torch.zeros(self.batch_size, device=self.device)
            total_reward = torch.zeros(self.batch_size, device=self.device)
            
            i = 0
            while torch.all(done == 0):

                # Get the action from the trained model
                with torch.no_grad():
                    mean_log_std = self.policy(state)
                    mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
                    
                    action = mean

                # Step the environment with the chosen action
                next_state, reward, done = env.step(action)
                # Store action
                actions[i,:,batch] = action.flatten()
                # Store reward (hedging error)
                rewards[:,batch] = reward
                # Accumulate the reward
                total_reward += reward
                # Move to the next state
                state = next_state
                
                i += 1
            
            loss = torch.sqrt(torch.mean(torch.square(torch.where(total_reward > 0, total_reward, 0))))
            total_val_reward[:,batch] = total_reward

            if render and batch % 100 == 0:
                print(f"Batch: {batch}/{batches-1}, Total Reward: {loss.item()}")
        rsmse = torch.sqrt(torch.mean(torch.square(torch.where(total_val_reward > 0, total_val_reward, 0))))

        return actions.flatten(1), rewards.flatten(), rsmse.item()
    

    def save(self, name):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'value_state_dict': self.value.state_dict()}, name)

    def load(self, name):
        checkpoint = torch.load(name, weights_only=True)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])