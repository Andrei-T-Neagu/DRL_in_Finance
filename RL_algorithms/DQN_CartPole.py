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

# Double DQN agent
class DoubleDQN:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma=0.99, epsilon=1.0, epsilon_min=0.05, lr=0.0001, batch_size=64, target_update=1, tau=0.1, double=True, dueling=True):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma                      # discount factor
        self.epsilon = epsilon                  # epsilon from epsilon-greedy action selection (random action taken with probability epsilon)
        self.epsilon_min = epsilon_min          # minimum value for epsilon
        self.lr = lr                            # learning rate 
        self.batch_size = batch_size            # batch size    
        self.target_update = target_update      # Frequency at which target model is updated
        self.double = double
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Main and target networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FFNN(in_features=state_size, out_features=action_size, num_layers=num_layers, hidden_size=hidden_size, dueling=dueling).to(self.device)
        self.model.apply(self.init_weights)
        self.target_model = FFNN(in_features=state_size, out_features=action_size, num_layers=num_layers, hidden_size=hidden_size, dueling=dueling).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.tau = tau
        # Synchronize target model with main model
        self.target_model.load_state_dict(self.model.state_dict())

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    # Set the parameters of the target model to be the same as the main model
    def update_target_model(self):
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_net_state_dict)

    # Store a tuple (s,a,r,s') in the replay memory buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Take an action according to the epsilon-greedy action selection strategy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.randint(self.action_size, size=(1,))
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values, dim=1)

    # perform an update based on a mini batch sampled from the replay memory buffer
    def replay(self):
        # do not perform an update if the replay memory buffer isn't filled enough to sample a batch
        if len(self.memory) < self.batch_size:
            return
        
        # sample a minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.vstack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.vstack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).squeeze(1)
        next_q_values = self.model(next_states).squeeze(1)
        with torch.no_grad():
            if self.double:
                next_q_target = self.target_model(next_states).squeeze(1)
            else:
                next_q_target = self.model(next_states).squeeze(1)

        # q_values of the actions from the minibatch
        q_value = q_values.gather(1, actions.unsqueeze(1))

        # pick the next action greedily using the main network
        next_action = torch.argmax(next_q_values, dim=1, keepdim=True)
        # q_values of the next actions using the target network
        target_q_value = next_q_target.gather(1, next_action)

        # expected q_values
        expected_q_value = rewards.unsqueeze(1) + (self.gamma * target_q_value * (1 - dones.unsqueeze(1)))

        # current q_value vs expected q_value
        loss = nn.HuberLoss()(q_value, expected_q_value)

        # perform update step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # load the model
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    # save the model
    def save(self, name):
        torch.save(self.model.state_dict(), name)

    # Training loop
    def train(self, env, episodes=200, lr_schedule = True):
        self.model.train()
        if self.double:
            self.target_model.train()
        
        episode_losses = []

        if lr_schedule:
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=episodes)

        epsilon_decay = (episodes-10)/episodes

        print("TRAINING DQN: ")

        for e in range(episodes):
            state, _ = env.reset()

            state = torch.Tensor([state]).to(self.device)
            done = torch.zeros(1)

            total_reward = torch.zeros(1, device=self.device)

            while torch.all(done == 0):

                action = self.act(state)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                
                next_state = torch.Tensor([next_state]).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                done = terminated or truncated
                done = torch.Tensor([done]).to(self.device)

                self.remember(state, action, reward, next_state, done)

                state = next_state

                total_reward += reward

                self.replay()

            episode_losses.append(total_reward.item())

            if self.double and e % self.target_update == 0:
                self.update_target_model()
            
            if lr_schedule and len(self.memory) > self.batch_size:
                self.scheduler.step()
            
            # decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= epsilon_decay
            
            print(f"Episode {e}/{episodes-1}, Epsilon: {self.epsilon}, Total Reward: {total_reward.item()}")
        
        return episode_losses
    
    # Testing loop
    def test(self, env, episodes=100):
        self.model.eval()

        test_losses = []

        print("TESTING DQN: ")

        for e in range(episodes):
            state, _ = env.reset()

            state = torch.Tensor([state]).to(self.device)
            done = torch.zeros(1)

            total_reward = torch.zeros(1, device=self.device)

            while torch.all(done == 0):

                with torch.no_grad():
                    q_values = self.model(state)
                    action = torch.argmax(q_values, dim=1)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                
                next_state = torch.Tensor([next_state]).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                done = terminated or truncated
                done = torch.Tensor([done]).to(self.device)

                state = next_state

                total_reward += reward

            test_losses.append(total_reward.item())

            print(f"Episode {e}/{episodes-1}, Total Reward: {total_reward.item()}")
        
        return np.mean(test_losses) 

num_layers = 3
nbs_units = 128
lr = 0.001
batch_size = 128
double = True
dueling = False

#for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

env = gym.make("CartPole-v1")
env.reset(seed=0)

dqn_agent = DoubleDQN(state_size=4, action_size=2, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size, double=double, dueling=dueling)
dqn_train_losses = dqn_agent.train(env, episodes=1000, lr_schedule=True)
test_losses = dqn_agent.test(env, episodes=100)
print("AVERAGE TEST LOSSES: ", test_losses)
ma_size = 20
hyperparameter_path = "/home/a_eagu/DRL_in_Finance/dqn_hyperparameters/"
dqn_losses = np.convolve(dqn_train_losses, np.ones(ma_size), 'valid') / ma_size
dqn_train_losses_fig = plt.figure(figsize=(12, 6))
plt.plot(dqn_train_losses, label="Total Reward")
plt.plot(dqn_losses, label="Total Reward (Moving Average)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.title("Total Reward " + str(ma_size) + " Episode Moving Average for DQN")
plt.savefig(hyperparameter_path + "training_losses/dqn_CARTPOLE_train_losses.png")
plt.close()
