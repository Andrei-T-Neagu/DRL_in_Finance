import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from neural_networks.FFNN import FFNN
from option_hedging.code_pytorch.DeepHedgingEnvironment import DeepHedgingEnvironment
import torch.optim.lr_scheduler as lr_scheduler

class DDPG:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma=0.99, lr=0.0001, batch_size=128, epochs=10, target_update=20, tau=0.5):
        self.state_size = state_size            
        self.action_size = action_size
        self.gamma = gamma                      # discount factor
        self.lr = lr                            # learning rate
        self.batch_size = batch_size            # batch size
        self.epochs = epochs                    # number of epochs
        self.target_update = target_update      # Frequency at which target model is updated
        # Experience replay buffer
        self.memory = deque(maxlen=10000)

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

    def get_action(self, state):
        # Predict mean and log_std
        action = self.policy(state)
        action = action + torch.randn(action.shape, device=self.device) * self.epsilon
        return action

    # perform an update based on a mini batch sampled from the replay memory buffer
    def replay(self):
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
            target_q_values = self.target_value(torch.cat([next_states, target_actions], dim=1))
            y = rewards + self.gamma * (1 - dones) * target_q_values
        q_values = self.value(torch.cat([states, actions], dim=1))

        # current q_value vs expected q_value
        value_loss = nn.HuberLoss()(q_values, y)

        # perform update step on value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # policy update
        policy_loss = -self.value(torch.cat([states, self.policy(states)], dim=1)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def train(self, env, val_env, episodes=200, lr_schedule = True):
        self.policy.train()
        self.value.train()
        
        env.train()
        val_env.test()

        episode_val_loss = []

        if lr_schedule:
            self.value_scheduler = lr_scheduler.LinearLR(self.value_optimizer, start_factor=1.0, end_factor=0.01, total_iters=episodes)
            self.policy_scheduler = lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.01, total_iters=episodes)

        self.epsilon = 1.0
        epsilon_decay = self.epsilon/episodes

        print("TRAINING DDPG: ")

        for e in range(episodes):
            state = env.reset()
            val_state = val_env.reset(self.batch_size)

            done = torch.zeros(1,1)
            val_done = torch.zeros(self.batch_size)

            total_reward = torch.zeros(1, device=self.device)
            val_total_reward = torch.zeros(self.batch_size, device=self.device)
            
            while torch.all(done == 0):

                with torch.no_grad():
                    action = self.get_action(state)
                    val_action = self.policy(val_state)

                next_state, reward, done = env.step(action)
                val_next_state, val_reward, val_done = val_env.step(val_action)

                reward = -torch.square(torch.where(reward > 0, reward, -0))
                
                self.remember(state, action, reward, next_state, done)

                state = next_state
                val_state = val_next_state

                total_reward += reward
                val_total_reward += val_reward

                self.replay()
            val_loss = torch.sqrt(torch.mean(torch.square(torch.where(val_total_reward > 0, val_total_reward, 0))))

            episode_val_loss.append(val_loss.item())

            if e % self.target_update == 0:
                self.update_target_models()
            
            if lr_schedule and len(self.memory) > self.batch_size:
                self.value_scheduler.step()
                self.policy_scheduler.step()
            
            self.epsilon -= epsilon_decay

            if e % 100 == 0:
                print(f"Episode {e}/{episodes-1}, Validation Loss: {val_loss.item()}")

        self.save("ddpg_model.pth")
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

        print("TESTING DDPG: ")
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

            if batch % 100 == 0:
                print(f"Batch: {batch}/{batches-1}, Total Reward: {loss.item()}")
        rsmse = torch.sqrt(torch.mean(torch.square(torch.where(total_val_reward > 0, total_val_reward, 0))))

        return actions.flatten(1), rewards.flatten(), rsmse.item()

    def save(self, name):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'value_state_dict': self.value.state_dict()}, name)

    def load(self, name):
        checkpoint = torch.load(name)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])