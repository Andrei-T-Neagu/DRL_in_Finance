import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from neural_networks.FFNN import FFNN
from option_hedging.DeepHedgingEnvironment import DeepHedgingEnvironment
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

class PG:
    """
    Policy Gradient RL algorithm with a deterministic policy
    
    Args:
    - state_size  | int   | size of the state space of the environment.
    - action_size | int   | size of the action space of the environment.
    - num_layers  | int   | number of the layers of the neural network (minimum of 2).
    - hidden_size | int   | number of neurons in each hidden layer of the neural network.
    - gamma       | float | return discount factor (0, 1].
    - lr          | float | learning rate.
    - batch_size  | int   | batch size for the environment and neural networks.

    Returns:
    - None
    """
    def __init__(self, config, state_size, action_size, gamma = 1.0, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = config.get("lr")                          # learning rate
        self.batch_size = config.get("batch_size")             # batch size
        self.num_layers = config.get("num_layers")
        self.hidden_size = config.get("hidden_size")
 
        self.gamma = gamma                      # Discount factor for the reward
        # Main and target networks
        self.device = device
        self.model = FFNN(in_features=state_size, out_features=action_size, num_layers=self.num_layers, hidden_size=self.hidden_size).to(self.device)
        self.model.apply(self.init_weights)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def init_weights(self, m):
        """
        Helper function which initializes the weights of the neural network using Xavier normal initialization. 
        To be called using model.apply(self.init_weights).

        Args:
        - m | torch.nn.Module | module for which the weights are initialized

        Returns: 
        - None
        """
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    # load the model
    def load(self, name):
        """
        Loads the previously saved model

        Args:
        - name | string | path and name of the saved model

        Returns:
        - None
        """
        self.model.load_state_dict(torch.load(name, weights_only=True))

    # save the model
    def save(self, name):
        """
        Save the current model

        Args:
        - name | string | path and name of the saved model

        Returns:
        - None
        """
        torch.save(self.model.state_dict(), name)

    # Training loop
    def train(self, env, val_env, BS_rsmse, episodes=1000, lr_schedule = True, render=False, path=None):
        """
        Training loop for policy gradient with a deterministic policy

        Args:
        - env            | DeepHedgingEnvironment | the deep hedging environment
        - episodes       | int                    | the number of episodes to train for
        - BS_rsmse       | float                  | rsmse achieved by Black-Scholes delta hedge
        - lr_schedule    | boolean                | whether to use a linear decay scheduler for the learning rate

        Returns:
        - episode_losses | list                   | a list of the episode total rewards/losses
        """
        self.model.train()
        env.train()
        val_env.train()

        episode_val_loss = []
        best_val_loss = 9999

        if lr_schedule:
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=episodes)
        
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

            # Compute validation losses
            if e % 1000 == 0:
                _, _, val_rsmse = self.test(val_env)
                self.model.train()
                episode_val_loss.append(val_rsmse)
                if val_rsmse < best_val_loss and path is not None:
                    self.save(path + "best_pg_model.pth")

            if render and e % 1000 == 0:
                print(f"Episode {e}/{episodes-1}, Validation RSMSE: {val_rsmse}")
        
            # Early stopping
            if len(episode_val_loss) > 5 and val_rsmse < BS_rsmse:
                if episode_val_loss[-6] < min(episode_val_loss[-5:]):
                    break

        return episode_val_loss

    def test(self, env, render=False):
        """
        Test a trained DQN agent on the environment.
        
        Args:
        - env     | DeepHedgingEnvironment         | The environment to test on.

        Returns:
        - actions | torch.Tensor[env.N, test_size] | actions performed on the test set
        - rewards | torch.Tensor[test_size]        | total rewards obtained for each path
        - rsmse   | float                          | root semi mean squared error of hedging losses
        """
        self.model.eval()
        env.test()
        set_size = env.dataset.shape[0]
        num_points = env.N
        batches = int(set_size/self.batch_size)

        actions = torch.zeros(num_points, self.batch_size, batches, device=self.device)
        rewards = torch.zeros(self.batch_size, batches, device=self.device)

        total_val_reward = torch.zeros(self.batch_size, batches, device=self.device)

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

            if render and batch % 100 == 0:
                print(f"Batch: {batch}/{batches-1}, Total Reward: {loss.item()}")
        rsmse = torch.sqrt(torch.mean(torch.square(torch.where(total_val_reward > 0, total_val_reward, 0))))
        return actions.flatten(1), rewards.flatten(), rsmse.item()
