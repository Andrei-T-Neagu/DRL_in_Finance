import sys
sys.path.insert(0,".")
import math
import torch
import torch.nn.functional as F

class DeepHedgingEnvironment():
    
    def __init__(self, nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, 
                 position_type, strike, V_0, prepro_stock, nbs_shares, light,
                 train_set, test_set, trans_costs = 0.0, discretized = False):
        
        self.nbs_point_traj = nbs_point_traj
        self.batch_size = 1
        self.r_borrow = r_borrow
        self.r_lend = r_lend
        self.S_0 = S_0
        self.T = T

        self.option_type = option_type
        self.position_type = position_type

        self.V_0 = V_0
        self.prepro_stock = prepro_stock
        self.nbs_shares = nbs_shares
        self.N = self.nbs_point_traj - 1   # number of time-steps
        self.dt = self.T / self.N       # time_step size

        self.strike = strike
        self.light = light
        self.trans_costs = trans_costs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.discretized = discretized                                                                      # For if the action space is discretized
        self.discretized_actions = torch.arange(start=-0.5, end=2.0, step=0.05, device=self.device)

        self.train_set = train_set.to(self.device)
        self.test_set = test_set.to(self.device)
        self.dataset = train_set.to(self.device)
        self.path = 0                   # the current path in the dataset

    def train(self):
        """
        Initializes the dataset to be the training set 
        
        Args:
        - None

        Returns:
        - None
        """
        self.dataset = self.train_set
        self.path = 0
        
    # Initialize the datasets for training
    def test(self):
        """
        Initializes the dataset to be the testing set 
        
        Args:
        - None

        Returns:
        - None
        """
        self.dataset = self.test_set
        self.path = 0

    # Initialize the environment variables
    def reset(self, batch_size=1):
        """
        Resets the Deep Hedging environment
        
        Args:
        - None

        Returns:
        - self.input_t: torch.Tensor of size [self.batch_size, 6] or [self.batch_size, 3] depending on the number of features. Features (state) to be input into the neural network.
        """
        self.t = 0                                                                  # the time step in the path
        self.batch_size = batch_size                                                # Set the batch size of the environment samples (size 1 if using replay memory buffer)
        self.done = torch.zeros(self.batch_size, device=self.device)                # variable which determines if the path (episode) is done

        self.delta_t = torch.zeros(self.batch_size, device=self.device)                         # torch.Tensor of size [self.batch_size], number of shares at each time step
        
        # Portfolio value prior to trading at each time-step
        # If long, you buy the option by borrowing V_0 from the bank
        if self.position_type == "long":
            self.V_t = -self.V_0 * torch.ones(self.batch_size, device=self.device)              # torch.Tensor of size [self.batch_size], portfolio value at time t              
        
        # If short, you receive the premium that you put in the bank
        elif self.position_type == "short":
            self.V_t = self.V_0 * torch.ones(self.batch_size, device=self.device)               # torch.Tensor of size [self.batch_size], portfolio value at time t 

        # Processing stock price
        if self.prepro_stock == "log":
            self.S_t = math.log(self.S_0) * torch.ones(self.batch_size, device=self.device)     # torch.Tensor of size [self.batch_size], normalization of the stock price
        elif self.prepro_stock == "log-moneyness":
            self.S_t = math.log(self.S_0 / self.strike) * torch.ones(self.batch_size, device=self.device)
        elif self.prepro_stock == "none":
            self.S_t = self.S_0 * torch.ones(self.batch_size, device=self.device)

        # Construct feature vector at the beginning of time t, S_t ad V_t are normalized 
        if self.light == True:
            self.input_t = torch.stack((self.dt * self.t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t), dim=1)
        else:
            self.input_t = torch.stack((self.dt * self.t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t, self.V_t/self.V_0), dim=1)
        
        return self.input_t

    def step(self, action):
        """
        Take an action in the environment and update the state variables and reward
        
        Args:
        - action: torch.Tensor of size [self.batch_size]. index of action to be taken for discretized actions, action to be taken for non-discretized actions.

        Returns:
        - self.input_t: torch.Tensor of size [batch_size, 6]. Features (state) to be input into the neural network
        - self.hedging_error: torch.Tensor of size [batch_size]. Hedging errors at each time step.
        - self.done: torch.Tensor of size [batch_size]. Either ones if last time step or zeros otherwise.
        """
        # un-normalize price
        self.S_t = self.inverse_processing(self.S_t)

        # When actions are discretized
        if self.discretized:
            self.delta_t_next = self.discretized_actions[action]                            # torch.Tensor of size [self.batch_size]. Action to be taken at next time step     
            self.delta_t_next = self.delta_t_next.flatten()
        else:
            self.delta_t_next = action.flatten()
        # Once the hedge is computed, update M_t (cash reserve)
        if self.t == 0:
            diff_delta_t = self.delta_t_next                                            # torch.Tensor of size [self.batch_size]. Difference in hedging position from last period
            cashflow = self.liquid_func(self.S_t, -diff_delta_t)    # torch.Tensor of size [self.batch_size]. Monetary gain or loss from last period.
            self.M_t = self.V_t + cashflow                                              # torch.Tensor of size [self.batch_size]. Time-t amount in the bank account (cash reserve)
        else:
            # Compute amount in cash reserve
            diff_delta_t = self.delta_t_next - self.delta_t
            cashflow = self.liquid_func(self.S_t, -diff_delta_t)
            self.M_t = self.int_rate_bank(self.M_t) + cashflow  # time-t amount in cash reserve

        # Update features for next time step (market impact persistence already updated)
        # Update stock price

        batch = self.dataset[self.path*self.batch_size:(self.path+1)*self.batch_size,:]     # torch.Tensor of size [self.batch_size, self.N] representing a batch of price paths
        self.S_t = batch[:,self.t+1]                                                        # torch.Tensor of size [self.batch_size] representing a batch of prices at the next time step
        
        # Liquidation portfolio value
        L_t = self.liquid_func(self.S_t, self.delta_t_next)     # torch.Tensor of size [self.batch_size]. Revenue from liquidating
        self.V_t = self.int_rate_bank(self.M_t) + L_t                               # torch.Tensor of size [self.batch_size]. Portfolio value.
        # Processing stock price
        if self.prepro_stock == "log":
            self.S_t = torch.log(self.S_t)
        elif self.prepro_stock == "log-moneyness":
            self.S_t = torch.log(self.S_t/self.strike)

        # input at the next time step
        if self.light == True:
            self.input_t = torch.stack((self.dt * self.t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t_next), dim=1)
        else:
            self.input_t = torch.stack((self.dt * self.t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t_next, self.V_t/self.V_0), dim=1)
        
        self.t += 1                                                             # iterate time step
        if self.t == self.N:                                                    # if t is the final time step
            self.done = torch.ones(self.batch_size, device=self.device)         #   set done to be True
            self.path += 1                                                      #   iterate path index
            self.t = 0                                                          #   set time step to 0
            if (self.path)*self.batch_size > self.dataset.shape[0]-1:           #   if path is the final path in the dataset
                self.path = 0                                                   #       reset path counter
            # Compute hedging error at maturity
            # Check if worth it to execute or not
            # Currently only working for call options
            self.M_t = self.int_rate_bank(self.M_t)         
            self.S_t = self.inverse_processing(self.S_t)
            
            # If call option: buyer executes iif profit selling > K 
            if self.position_type == "short":
                if self.option_type == "call":
                    self.condition = torch.where(self.cost_selling(self.S_t, self.nbs_shares) >= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_selling(self.S_t, self.nbs_shares) >= self.nbs_shares * self.strike, 
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next - self.nbs_shares) + self.nbs_shares * self.strike, 
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next))
                if self.option_type == "put":
                    self.condition = torch.where(self.cost_buying(self.S_t, self.nbs_shares) <= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_buying(self.S_t, self.nbs_shares) <= self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next + self.nbs_shares) - self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next))
            if self.position_type == "long":
                if self.option_type == "call":
                    self.condition = torch.where(self.cost_selling(self.S_t, self.nbs_shares) >= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_selling(self.S_t, self.nbs_shares) >= self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next + self.nbs_shares) - self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next))
                if self.option_type == "put":
                    self.condition = torch.where(self.cost_buying(self.S_t, self.nbs_shares) <= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_buying(self.S_t, self.nbs_shares) <= self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next - self.nbs_shares) + self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next))
            self.hedging_error = -self.hedging_gain
            
        else:
            self.hedging_error = torch.zeros(self.batch_size, device=self.device)
            self.done = torch.zeros(self.batch_size, device=self.device)

        # update delta_t
        self.delta_t = self.delta_t_next

        return self.input_t, self.hedging_error, self.done

    # Reverse the processing of the stock price
    def inverse_processing(self, paths):
        if (self.prepro_stock == "log"):
            paths = torch.exp(paths)
        elif (self.prepro_stock == "log-moneyness"):
            paths = self.strike * torch.exp(paths)
        return paths
    
    # Profit from selling
    # Inputs:
    #   - S_t: stock price
    #   - x: number of shares
    # Returns:
    #   - Profit from selling
    def cost_selling(self, S_t, x):
        return (S_t * x) * (1.0 - self.trans_costs)
    
    # Cost of buying
    # Inputs:
    #   - S_t: stock price
    #   - x: number of shares
    # Returns:
    #   - Cost of buying
    def cost_buying(self, S_t, x):
        return (S_t * x) * (1.0 + self.trans_costs)

    # Liquidation value L_t
    # Inputs:
    #   - S_t: stock price
    #   - x: number of shares
    # Returns:
    #   - Liquidation value
    def liquid_func(self, S_t, x):
        return self.cost_selling(S_t, F.relu(x)) - self.cost_buying(S_t, F.relu(-x))
    
    # Computation of bank interest (periodic)
    def int_rate_bank(self, x):
        return F.relu(x) * (1 + self.r_lend) ** self.dt - F.relu(-x) * (1 + self.r_borrow) ** self.dt

    # Returns the loss computed on the hedging errors
    def loss(self):
        if self.loss_type == "RMSE":
            loss = torch.sqrt(torch.mean(torch.square(self.hedging_error)))
        elif self.loss_type == "RMSE per share":
            loss = torch.sqrt(torch.mean(torch.square(self.hedging_error))) / self.nbs_shares
        elif self.loss_type == "RSMSE":
            loss = torch.sqrt(torch.mean(torch.square(torch.where(self.hedging_error > 0, self.hedging_error, 0))))
        elif self.loss_type == "RSMSE per share":
            loss = torch.sqrt(torch.mean(torch.square(torch.where(self.hedging_error > 0, self.hedging_error, 0)))) / self.nbs_shares
        return loss