import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from neural_networks.FFNN import FFNN
from option_hedging.code_pytorch.DeepHedgingEnvironment import DeepHedgingEnvironment

# Double DQN agent
class DoubleDQN:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99998, lr=0.001, batch_size=64, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma                      # discount factor
        self.epsilon = epsilon                  # epsilon from epsilon-greedy action selection (random action taken with probability epsilon)
        self.epsilon_min = epsilon_min          # minimum value for epsilon
        self.epsilon_decay = epsilon_decay      # decay rate of epsilon
        self.lr = lr                            # learning rate 
        self.batch_size = batch_size            # batch size    

        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        
        # Main and target networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FFNN(in_features=state_size, out_features=action_size, num_layers=num_layers, hidden_size=hidden_size).to(self.device)
        self.target_model = FFNN(in_features=state_size, out_features=action_size, num_layers=num_layers, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Synchronize target model with main model
        self.update_target_model()

    # Set the parameters of the target model to be the same as the main model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Store a tuple (s,a,r,s') in the replay memory buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Take an action according to the epsilon-greedy action selection strategy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.randint(self.action_size, size=(1,))
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmin(action_values, dim=1)

    # perform an update based on a mini batch sampled from the replay memory buffer
    def replay(self):
        # do not perform an update if the replay memory buffer isn't filled enough to sample a batch
        if len(self.memory) < self.batch_size:
            return
        
        # sample a minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.stack(list(states), dim=0).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.stack(list(next_states), dim=0).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).squeeze(1)
        next_q_values = self.model(next_states).squeeze(1)
        next_q_target = self.target_model(next_states).squeeze(1)

        # q_values of the actions picked in the minibatch
        q_value = q_values.gather(1, actions)

        # pick the next action greedily
        next_action = torch.argmin(next_q_values, dim=1).unsqueeze(1)
        # q_values of the next actions picked in the minibatch
        target_q_value = next_q_target.gather(1, next_action)

        # expected (target) q_values
        expected_q_value = rewards + (self.gamma * target_q_value * (1 - dones))

        # current q_value vs target q_value
        loss = nn.MSELoss()(q_value, expected_q_value)

        # perform update step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # load the model
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    # save the model
    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Training loop
def train_dqn(model, env, episodes=200, target_update_freq=10):
    agent = model
    agent.model.train()
    env.train()

    print("TRAINING DQN: ")

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward = torch.square(torch.where(reward > 0, reward, 0))/100
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward[0].item()
            
            agent.replay()
        
        if e % target_update_freq == 0:
            agent.update_target_model()
        
        print(f"Episode {e+1}/{episodes}, Epsilon: {agent.epsilon}, Total Reward: {total_reward}")

    agent.save("dqn_model.pth")

def test_dqn(model, env, episodes=100, batch_size=512):
    """
    Test a trained DQN agent on the environment.
    
    Args:
    - env: The environment to test on.
    - agent: The trained DQN agent.
    - episodes: Number of episodes to test the agent.
    
    Returns:
    - total_rewards: List of total rewards for each episode.
    """
    env.test()

    print("TESTING DQN: ")

    for episode in range(episodes):
        state = env.reset(batch_size)  # Initialize the environment and get the initial state
        done = torch.zeros(batch_size)
        total_reward = torch.zeros(batch_size)

        agent = model
        agent.model.eval()
        
        while torch.all(done == 0):

            # Get the action from the trained model (greedy policy, no epsilon-greedy)
            with torch.no_grad():
                q_values = agent.model(state)
                action = torch.argmin(q_values, dim=1)

            # Step the environment with the chosen action
            next_state, reward, done = env.step(action)
            # Accumulate the reward
            total_reward = total_reward + reward.cpu()
            # Move to the next state
            state = next_state
        
        rsmse = torch.sqrt(torch.mean(torch.square(torch.where(total_reward > 0, total_reward, 0)))).item()
        print(f"Episode {episode+1}/{episodes}, rsmse: {rsmse}")