import sys
sys.path.insert(0,".")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from neural_networks.FFNN import FFNN
from neural_networks.CategoricalActor import CategoricalFFNN

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')

SEED = 1234

train_env.reset(seed=SEED);
test_env.reset(seed=SEED+1);
np.random.seed(SEED);
torch.manual_seed(SEED);

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred
    
INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n

policy = CategoricalFFNN(INPUT_DIM, OUTPUT_DIM, num_layers=2, hidden_size=HIDDEN_DIM)
critic = FFNN(INPUT_DIM, 1, num_layers=2, hidden_size=HIDDEN_DIM)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

policy.apply(init_weights)
critic.apply(init_weights)

LEARNING_RATE = 0.01

# optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

optimizer = optim.Adam([
                       {"params": policy.parameters(), "lr": LEARNING_RATE},
                       {"params": critic.parameters(), "lr": LEARNING_RATE}
                       ])

policy_optimizer = optim.Adam(policy.parameters(), LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), LEARNING_RATE)

def get_action(state):
    # Predict mean and log_std
    dist = policy(state)
    action = dist.sample()
    action_log_prob = dist.log_prob(action)
    return action, action_log_prob

def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):
        
    policy.train()
    critic.train()

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()[0]

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)

        #append state here, not after we get the next state from env.step()
        states.append(state)
        
        value_pred = critic(state)
        
        action, log_prob_action = get_action(state)
        
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward
    
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    
    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        value_pred = critic(states)
        value_pred = value_pred.squeeze(-1)
        dist = policy(states)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss, total_value_loss, episode_reward

def calculate_returns(rewards, discount_factor, normalize = True):
    
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

def calculate_advantages(returns, values, normalize = True):
    
    advantages = returns - values
    
    if normalize:
        
        advantages = (advantages - advantages.mean()) / advantages.std()
        
    return advantages

def evaluate(env, policy):
    
    policy.eval()
    
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()[0]

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
        
            action_prob = policy(state)
                
        action = action_prob.sample()
                
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward
        
    return episode_reward

MAX_EPISODES = 500
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25

PRINT_EVERY = 1
PPO_STEPS = 5
PPO_CLIP = 0.2

train_rewards = []
test_rewards = []

for episode in range(1, MAX_EPISODES+1):
    
    policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
    
    test_reward = evaluate(test_env, policy)
    
    train_rewards.append(train_reward)
    test_rewards.append(test_reward)
    
    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    
    if episode % PRINT_EVERY == 0:
    
        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')