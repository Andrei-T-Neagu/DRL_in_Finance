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
class DoubleDQN:
    def __init__(self, state_size, action_size, num_layers, hidden_size, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9997, lr=0.0001, batch_size=128, target_update=20, tau=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma                      # discount factor
        self.epsilon = epsilon                  # epsilon from epsilon-greedy action selection (random action taken with probability epsilon)
        self.epsilon_min = epsilon_min          # minimum value for epsilon
        self.epsilon_decay = epsilon_decay      # decay rate of epsilon
        self.lr = lr                            # learning rate 
        self.batch_size = batch_size            # batch size    
        self.target_update = target_update      # Frequency at which target model is updated
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Main and target networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FFNN(in_features=state_size, out_features=action_size, num_layers=num_layers, hidden_size=hidden_size, dueling=True).to(self.device)
        self.target_model = FFNN(in_features=state_size, out_features=action_size, num_layers=num_layers, hidden_size=hidden_size, dueling=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.tau = tau
        # Synchronize target model with main model
        self.target_model.load_state_dict(self.model.state_dict())

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
            next_q_target = self.target_model(next_states).squeeze(1)

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
        # for p in self.model.parameters():
        #     print(p.grad.norm())
        # nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

    # load the model
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    # save the model
    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Training loop
    def train(self, env, val_env, episodes=200, lr_schedule = True):
        self.model.train()
        env.train()
        val_env.test()

        episode_val_loss = []

        if lr_schedule:
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=episodes)

        print("TRAINING DQN: ")

        for e in range(episodes):
            state = env.reset()
            val_state = val_env.reset(self.batch_size)

            done = torch.zeros(1)
            val_done = torch.zeros(self.batch_size)

            total_reward = torch.zeros(1, device=self.device)
            val_total_reward = torch.zeros(self.batch_size, device=self.device)
            
            while torch.all(done == 0):

                action = self.act(state)
                with torch.no_grad():
                    val_q_values = self.model(val_state)
                    val_action = torch.argmax(val_q_values, dim=1)

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
                self.update_target_model()
            
            if lr_schedule:
                self.scheduler.step()
            
            # decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if e % 100 == 0:
                print(f"Episode {e}/{episodes-1}, Epsilon: {self.epsilon}, Validation Loss: {val_loss.item()}")

        self.save("dqn_model.pth")
        return episode_val_loss

    def test(self, env):
        """
        Test a trained DQN agent on the environment.
        
        Args:
        - env: The environment to test on.
        - episodes: Number of episodes to test the agent.
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

        print("TESTING DQN: ")

        for batch in range(batches):
            state = env.reset(self.batch_size)  # Initialize the environment and get the initial state
            done = torch.zeros(self.batch_size)
            total_reward = torch.zeros(self.batch_size, device=self.device)

            i = 1
            while torch.all(done == 0):

                # Get the action from the trained model (greedy policy, no epsilon-greedy)
                with torch.no_grad():
                    q_values = self.model(state)
                    action = torch.argmax(q_values, dim=1)

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
                print(f"Episode {batch}/{batches-1}, Total Reward: {loss.item()}")
        rsmse = torch.sqrt(torch.mean(torch.square(torch.where(total_val_reward > 0, total_val_reward, 0))))
        return actions.flatten(1), rewards.flatten(), rsmse.item()