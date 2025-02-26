import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from neural_networks.FFNN import FFNN
from option_hedging.DeepHedgingEnvironment import DeepHedgingEnvironment
import torch.optim.lr_scheduler as lr_scheduler

class DDPG:
    def __init__(self, config=None, state_size=3, action_size=1, twin_delayed=False, device='cpu'):
        self.state_size = state_size            
        self.action_size = action_size
        
        self.gamma = 1.0                                           # discount factor
        
        self.lr = config.get("lr", 0.0001)                          # learning rate
        self.batch_size = config.get("batch_size", 128)             # batch size
        self.num_layers = config.get("num_layers")
        self.hidden_size = config.get("hidden_size")
        
        self.twin_delayed = twin_delayed
        self.target_update = 2 if self.twin_delayed else 1          # Frequency at which target model is updated
        self.tau = 0.1
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)

        self.device = device
        # Policy network
        self.policy = FFNN(state_size, action_size, self.num_layers, self.hidden_size, policy=True).to(self.device)
        self.target_policy = FFNN(state_size, action_size, self.num_layers, self.hidden_size, policy=True).to(self.device)
        
        # Value network
        self.value = FFNN(state_size + 1, 1, self.num_layers, self.hidden_size, value=False).to(self.device)
        self.target_value = FFNN(state_size + 1, 1, self.num_layers, self.hidden_size, value=False).to(self.device)

        self.policy.apply(self.init_weights)
        self.value.apply(self.init_weights)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), self.lr)

        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())

        if self.twin_delayed:
            self.value2 = FFNN(state_size + 1, 1, self.num_layers, self.hidden_size, value=False).to(self.device)
            self.value2.apply(self.init_weights)
            self.value2_optimizer = optim.Adam(self.value2.parameters(), self.lr)
            self.target_value2 = FFNN(state_size + 1, 1, self.num_layers, self.hidden_size, value=False).to(self.device)
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
        action = action + torch.randn(action.shape, device=self.device) * self.epsilon
        action = torch.clamp(action, -0.5, 2.0)
        return action

    # perform an update based on a mini batch sampled from the replay memory buffer
    def replay(self, e):
        # do not perform an update if the replay memory buffer isn't filled enough to sample a batch
        if len(self.memory) < self.batch_size:
            return
        
        # sample a minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.vstack(states)
        actions = torch.vstack(actions)
        rewards = torch.vstack(rewards)
        next_states = torch.vstack(next_states)
        dones = torch.vstack(dones)
    
        with torch.no_grad():
            target_actions = self.target_policy(next_states) 
            if self.twin_delayed:
                noise = torch.randn(target_actions.shape, device=self.device) * self.epsilon
                clipped_noise = torch.clamp(noise, -0.5, 0.5)
                target_actions = torch.clamp(target_actions + clipped_noise, -0.5, 2.0)
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
            value2_loss = nn.MSELoss()(q_values2, y)

        # current q_value vs expected q_value
        value_loss = nn.MSELoss()(q_values, y)

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
                policy_loss = -self.value(torch.cat([states, self.policy(states)], dim=1)).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
        else:
            # policy update
                policy_loss = -self.value(torch.cat([states, self.policy(states)], dim=1)).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
    def train(self, env, val_env, BS_rsmse, episodes=1000, lr_schedule = True, render=False, path=None):
        self.policy.train()
        self.value.train()
        if self.twin_delayed:
            self.value2.train()
        
        env.train()
        val_env.train()

        episode_val_loss = []
        best_val_loss = 9999

        if lr_schedule:
            self.value_scheduler = lr_scheduler.LinearLR(self.value_optimizer, start_factor=1.0, end_factor=0.0, total_iters=episodes)
            if self.twin_delayed:
                self.value2_scheduler = lr_scheduler.LinearLR(self.value2_optimizer, start_factor=1.0, end_factor=0.0, total_iters=episodes)
            self.policy_scheduler = lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=episodes)

        self.epsilon = 0.5
        epsilon_decay = self.epsilon/(episodes+1)

        print("TRAINING DDPG: ")

        for e in range(episodes):
            state = env.reset()

            done = torch.zeros(1,1)

            total_reward = torch.zeros(1, device=self.device)
            while torch.all(done == 0):

                with torch.no_grad():
                    action = self.get_action(state)

                next_state, reward, done = env.step(action)

                reward = -torch.square(torch.where(reward > 0, reward, -0))
                
                self.remember(state, action, reward, next_state, done)

                state = next_state

                total_reward += reward

                self.replay(e)

            if e % self.target_update == 0:
                self.update_target_models()
            
            if lr_schedule and len(self.memory) > self.batch_size:
                self.value_scheduler.step()
                if self.twin_delayed:
                    self.value2_scheduler.step()
                self.policy_scheduler.step()
            
            self.epsilon -= epsilon_decay

            # Compute validation losses
            if e % 1000 == 0:
                _, _, val_rsmse = self.test(val_env)
                self.policy.train()
                episode_val_loss.append(val_rsmse)
                if val_rsmse < best_val_loss and path is not None:
                    self.save(path + "best_ddpg_model.pth")

            if render and e % 10000 == 0:
                print(f"Episode {e}/{episodes-1}, Validation Loss: {val_rsmse}")
            
            # Early stopping
            if len(episode_val_loss) > 5 and val_rsmse < BS_rsmse:
                if episode_val_loss[-6] < min(episode_val_loss[-5:]):
                    break

        return episode_val_loss

    def test(self, env, render=False):
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
                    action = self.policy(state)

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
        if self.twin_delayed:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                        'value_state_dict': self.value.state_dict(),
                        'value2_state_dict': self.value2.state_dict(),}, name)
        else:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                        'value_state_dict': self.value.state_dict(),}, name)
            
    def load(self, name):
        checkpoint = torch.load(name, weights_only=True)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())

        if self.twin_delayed:
            self.value2.load_state_dict(checkpoint['value2_state_dict'])
            self.target_value2.load_state_dict(self.value2.state_dict())