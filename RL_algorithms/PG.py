import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from neural_networks.FFNN import FFNN
from option_hedging.code_pytorch.DeepHedgingEnvironment import DeepHedgingEnvironment
import torch.optim.lr_scheduler as lr_scheduler

# Double DQN agent
class PG:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma = 1.0, lr=0.0001, batch_size=512):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr                            # learning rate 
        self.batch_size = batch_size            # batch size    
        self.gamma = gamma                      # Discount factor for the reward
        # Main and target networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FFNN(in_features=state_size, out_features=action_size, num_layers=num_layers, hidden_size=hidden_size).to(self.device)
        self.model.apply(self.init_weights)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    # load the model
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    # save the model
    def save(self, name):
        torch.save(self.model.state_dict(), name)

    # Training loop
    def train(self, env, episodes=1000, lr_schedule = True):
        self.model.train()
        env.train()
        
        if lr_schedule:
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0001, total_iters=episodes)
        
        print("TRAINING PG: ")
        for e in range(episodes):
            state = env.reset(self.batch_size)
            done = torch.zeros(self.batch_size)
            total_reward = torch.zeros(self.batch_size, device=self.device)
            i = 0

            while torch.all(done == 0):
                action = self.model(state)
                next_state, reward, done = env.step(action)
                state = next_state

                total_reward += (self.gamma ** i) * reward
                i += 1

            loss = torch.sqrt(torch.mean(torch.square(torch.where(total_reward > 0, total_reward, 0))))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if lr_schedule:
                self.scheduler.step()
            
            if e % 100 == 0:
                print(f"Episode {e}/{episodes-1}, Total Reward: {loss.item()}")

        # self.save("pg_model.pth")

    def test(self, env):
        """
        Test a trained DQN agent on the environment.
        
        Args:
        - env: The environment to test on.
        - agent: The trained DQN agent.
        - batch_size: Batch size.

        Returns:
        - total_rewards: List of total rewards for each episode.
        """
        self.model.eval()
        env.test()
        train_size = env.dataset.shape[0]
        num_points = env.N
        batches = int(train_size/self.batch_size)

        actions = torch.zeros(num_points, self.batch_size, batches, device=self.device)
        rewards = torch.zeros(self.batch_size, batches, device=self.device)

        total_val_reward = torch.zeros(self.batch_size, batches, device=self.device)

        print("TESTING PG: ")
        for batch in range(batches):
            state = env.reset(self.batch_size)  # Initialize the environment and get the initial state
            done = torch.zeros(self.batch_size)
            total_reward = torch.zeros(self.batch_size, device=self.device)

            i = 0
            while torch.all(done == 0):
                # Get the action from the trained model (greedy policy, no epsilon-greedy)
                with torch.no_grad():
                    action = self.model(state)
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

            if batch % 100 == 0:
                print(f"Batch: {batch}/{batches-1}, Total Reward: {loss.item()}")
        rsmse = torch.sqrt(torch.mean(torch.square(torch.where(total_val_reward > 0, total_val_reward, 0))))

        return actions.flatten(1), rewards.flatten(), rsmse.item()