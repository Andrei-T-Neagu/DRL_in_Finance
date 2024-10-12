import datetime as dt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Utils_general
import sys
sys.path.insert(0,".")
import torch.optim.lr_scheduler as lr_scheduler
from neural_networks.FFNN import FFNN
from neural_networks.LSTM import LSTM_multilayer_cell
from neural_networks.GRU import GRU_multilayer_cell
from neural_networks.Transformer import Transformer

class DeepHedgingEnvironment():
    
    def __init__(self, model_type, nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                 nbs_shares, lambdas, light, train_set, test_set, name='model'):
        
        self.model_type = model_type
        self.nbs_point_traj = nbs_point_traj
        self.batch_size = 1
        self.r_borrow = r_borrow
        self.r_lend = r_lend
        self.stock_dyn = stock_dyn
        self.S_0 = S_0
        self.T = T
        self.alpha = alpha  # liquidity impact factor when buying
        self.beta = beta    # liquidity impact factor when selling
        self.loss_type = loss_type
        self.option_type = option_type
        self.position_type = position_type

        self.V_0 = V_0
        self.nbs_layers = nbs_layers
        self.nbs_units = nbs_units
        self.lr = lr
        self.prepro_stock = prepro_stock
        self.nbs_shares = nbs_shares
        self.N = self.nbs_point_traj - 1   # number of time-steps
        self.dt = self.T / self.N       # time_step size

        self.A_0 = 0    #initial value of persistence impact for the ask
        self.lambda_a = lambdas[0]    #persistence parameter for the ask
        self.B_0 = 0    #initial value of persistence impact for the bid
        self.lambda_b = lambdas[1]    #persistence parameter for the bid
        self.params_vect = params_vect
        self.strike = strike
        self.light = light
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discretized_actions = torch.arange(start=-0.5, end=2.0, step=0.05, device=self.device)

        self.train_set = train_set
        self.test_set = test_set
        self.dataset = train_set
        self.path = 0                   # the current path in the dataset

        # # Assign model type
        # if self.model_type == "FFNN":
        #     self.model = FFNN(in_features=6, num_layers=nbs_layers, hidden_size=nbs_units, dropout=dropout).to(self.device)
        # elif self.model_type == "LSTM":
        #     self.model = LSTM_multilayer_cell(batch_size=self.batch_size, input_size=6, hidden_size=self.nbs_units, num_layers=self.nbs_layers, device=self.device, dropout=dropout).to(self.device)
        # elif self.model_type == "GRU":
        #     self.model = GRU_multilayer_cell(batch_size=self.batch_size, input_size=6, hidden_size=self.nbs_units, num_layers=self.nbs_layers, device=self.device, dropout=dropout).to(self.device)
        # elif self.model_type == "Transformer":
        #     self.model = Transformer(in_features=6, seq_length=self.N, d_model=self.nbs_units, dim_feedforward=self.nbs_units, n_heads=self.num_heads, num_layers=self.nbs_layers, dropout=dropout).to(self.device)
        
        self.name = name
        print("Initial value of the portfolio: ", V_0)

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
        self.t = 0                      # the time step in the path
        self.done = False               # variable which determines if the path (episode) is done
        self.batch_size = batch_size    # Set the batch size of the environment samples (size 1 if using replay memory buffer)

        # Different models' features
        # if self.model_type == "LSTM" or self.model_type == "GRU":
        #     self.hs = [torch.zeros(self.batch_size, self.nbs_units, device=self.device) for i in range(self.nbs_layers)]
        # if self.model_type == "LSTM":
        #     self.cs = [torch.zeros(self.batch_size, self.nbs_units, device=self.device) for i in range(self.nbs_layers)]
        # if self.model == "Transformer":
        #     # padding mask that is passed to the transformer
        #     self.input_t_seq_mask = torch.ones(self.batch_size, self.N, device=self.device).type(torch.bool)

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
        
        self.A_t = self.A_0 * torch.ones(self.batch_size, device=self.device)                   # torch.Tensor of size [self.batch_size], current market impact parameters
        self.B_t = self.B_0 * torch.ones(self.batch_size, device=self.device)

        # Construct feature vector at the beginning of time t, S_t ad V_t are normalized 
        if self.light == True:
            self.input_t = torch.stack((self.dt * self.t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t), dim=1)
        else:
            self.input_t = torch.stack((self.dt * self.t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t, self.V_t/self.V_0, self.A_t, self.B_t), dim=1)

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
        
        # output of the model
        # if self.model_type == "FFNN":
        #     self.delta_t_next = self.model(input_t)
        # elif self.model_type == "LSTM":
        #     self.delta_t_next, hs, cs = self.model(input_t, hs, cs)
        # elif self.model_type == "GRU":
        #     self.delta_t_next, hs = self.model(input_t, hs)
        # elif self.model_type == "Transformer":
        #     # input without the padding
        #     if t == 0:
        #         input_t_concat = input_t.unsqueeze(dim=1)
        #     else:
        #         input_t_concat = torch.cat((input_t_concat, input_t.unsqueeze(dim=1)), dim=1)
        #     padding = torch.zeros(self.batch_size, self.N-(t+1), 6, device=self.device)
        #     # input sequence with the padding passed to the transformer
        #     input_t_seq = torch.cat((input_t_concat, padding), dim=1)
        #     # Update padding mask
        #     input_t_seq_mask[:, t] = False

        #     self.delta_t_next = self.model(input_t_seq, input_t_seq_mask)

        # When actions are discretized
        self.delta_t_next = self.discretized_actions[action]                            # torch.Tensor of size [self.batch_size]. Action to be taken at next time step     
        # Once the hedge is computed, update M_t (cash reserve)
        if self.t == 0:
            diff_delta_t = self.delta_t_next                                            # torch.Tensor of size [self.batch_size]. Difference in hedging position from last period
            cashflow = self.liquid_func(self.S_t, -diff_delta_t, self.A_t, self.B_t)    # torch.Tensor of size [self.batch_size]. Monetary gain or loss from last period.
            self.M_t = self.V_t + cashflow                                              # torch.Tensor of size [self.batch_size]. Time-t amount in the bank account (cash reserve)
        else:
            # Compute amount in cash reserve
            diff_delta_t = self.delta_t_next - self.delta_t
            cashflow = self.liquid_func(self.S_t, -diff_delta_t, self.A_t, self.B_t)
            self.M_t = self.int_rate_bank(self.M_t) + cashflow  # time-t amount in cash reserve

        # Compute liquidity impact and persistence
        # For ask:
        if self.lambda_a == -1:
            self.A_t = torch.zeros(self.batch_size, device=self.device)
        else:
            impact_ask = torch.where(diff_delta_t > 0, diff_delta_t, 0.0)
            self.A_t = (self.A_t + impact_ask) * math.exp(-self.lambda_a * self.dt)
        # For bid:
        if self.lambda_b == -1:
            self.B_t = torch.zeros(self.batch_size, device=self.device)
        else:
            impact_bid = torch.where(diff_delta_t < 0, -diff_delta_t, 0.0)
            self.B_t = (self.B_t + impact_bid) * math.exp(-self.lambda_b * self.dt)

        # Update features for next time step (market impact persistence already updated)
        # Update stock price

        batch = self.dataset[self.path*self.batch_size:(self.path+1)*self.batch_size,:]     # torch.Tensor of size [self.batch_size, self.N] representing a batch of price paths
        self.S_t = batch[:,self.t+1].to(device=self.device)                                 # torch.Tensor of size [self.batch_size] representing a batch of prices at the next time step

        # Liquidation portfolio value
        L_t = self.liquid_func(self.S_t, self.delta_t_next, self.A_t, self.B_t)     # torch.Tensor of size [self.batch_size]. Revenue from liquidating
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
            self.input_t = torch.stack((self.dt * self.t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t_next, self.V_t/self.V_0, self.A_t, self.B_t), dim=1)
        
        self.t += 1                                                             # iterate time step
        if self.t == self.N-1:                                                  # if t is the final time step
            self.done = torch.ones(self.batch_size)                             #   set done to be True
            self.path += 1                                                      #   iterate path index
            self.t = 0                                                          #   set time step to 0
            if (self.path+1)*self.batch_size > self.dataset.shape[0]-1:        #   if path is the final path in the dataset
                self.path = 0                                                   #       reset path counter
            # Compute hedging error at maturity
            # Check if worth it to execute or not
            # Currently only working for call options
            self.M_t = self.int_rate_bank(self.M_t)         
            self.S_t = self.inverse_processing(self.S_t)
            
            # If call option: buyer executes iif profit selling > K 
            if self.position_type == "short":
                if self.option_type == "call":
                    self.condition = torch.where(self.cost_selling(self.S_t, self.nbs_shares, self.B_t) >= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_selling(self.S_t, self.nbs_shares, self.B_t) >= self.nbs_shares * self.strike, 
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next - self.nbs_shares, self.A_t, self.B_t) + self.nbs_shares * self.strike, 
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next, self.A_t, self.B_t))
                if self.option_type == "put":
                    self.condition = torch.where(self.cost_buying(self.S_t, self.nbs_shares, self.B_t) <= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_buying(self.S_t, self.nbs_shares, self.A_t) <= self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next + self.nbs_shares, self.A_t, self.B_t) - self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next, self.A_t, self.B_t))
            if self.position_type == "long":
                if self.option_type == "call":
                    self.condition = torch.where(self.cost_selling(self.S_t, self.nbs_shares, self.B_t) >= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_selling(self.S_t, self.nbs_shares, self.B_t) >= self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next + self.nbs_shares, self.A_t, self.B_t) - self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next, self.A_t, self.B_t))
                if self.option_type == "put":
                    self.condition = torch.where(self.cost_buying(self.S_t, self.nbs_shares, self.B_t) <= self.nbs_shares * self.strike, 1, 0)
                    self.hedging_gain = torch.where(self.cost_buying(self.S_t, self.nbs_shares, self.A_t) <= self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next - self.nbs_shares, self.A_t, self.B_t) + self.nbs_shares * self.strike,
                                                    self.M_t + self.liquid_func(self.S_t, self.delta_t_next, self.A_t, self.B_t))
            self.hedging_error = -self.hedging_gain
            
            # print("HEDGING ERROR SHAPE: ", self.hedging_error.shape)
            # print("HEDGING ERROR: ", self.hedging_error)
        else:
            self.hedging_error = torch.zeros(self.batch_size)
            self.done = torch.zeros(self.batch_size)

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
    
    # Profit from selling (F_t^b)
    # Inputs:
    #   - S_t: stock price
    #   - x: number of shares
    #   - y: impact persistence for the bid
    # Returns:
    #   - Profit from selling
    def cost_selling(self, S_t, x, y):
        return S_t * ((1 + x + y) ** self.beta - (1 + y) ** self.beta)
    
    # Cost of buying (F_t^a)
    # Inputs:
    #   - S_t: stock price
    #   - x: number of shares
    #   - y: impact persistence for the ask
    # Returns:
    #   - Cost of buying
    def cost_buying(self, S_t, x, y):
        return S_t * ((1 + x + y) ** self.alpha - (1 + y) ** self.alpha)

    # Liquidation value L_t
    # Inputs:
    #   - S_t: stock price
    #   - x: number of shares
    #   - A_t: impact persistence for the ask
    #   - B_t: impact persistence for the bid
    # Returns:
    #   - Liquidation value
    def liquid_func(self, S_t, x, A_t, B_t):
        return self.cost_selling(S_t, F.relu(x), B_t) - self.cost_buying(S_t, F.relu(-x), A_t)
    
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

    def training(self, train_set, train_size, epochs, lr_schedule = True):
        start = dt.datetime.now()  # compute time
        self.losses_epochs = np.array([])
        best_loss = 99999999
        epoch = 0
        maxAt = np.array([])
        maxBt = np.array([])
        all_losses = np.array([])
        worse_loss = 0
        early_stop = False

        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if lr_schedule:
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)

        # Loop while we haven't reached the max epoch and early stopping criteria is not reached
        while (epoch < epochs):
            hedging_error_train = np.array([])
            strat = np.array([])
            exercised = np.array([])
            losses = np.array([])

            # mini batch training
            for i in range(int(train_size/self.batch_size)):
                
                if i % 100 == 0:
                    print("BATCH: " + str(i) + "/" + str(int(train_size/self.batch_size)))

                # Zero out gradients
                self.optimizer.zero_grad()
                
                batch = train_set[i*self.batch_size:(i+1)*self.batch_size,:]

                # Perform action on batch
                hedging_error, strategy, S_t_tensor, V_t_tensor, A_t_tensor, B_t_tensor = self.simulate_batch(batch=batch)
                
                # Compute and backprop loss
                loss = self.loss()
                loss.backward()

                # print(self.S_t_tensor.grad)

                # Take gradient step
                self.optimizer.step()
                
                all_losses = np.append(all_losses, loss.detach().cpu().numpy())
                losses = np.append(losses, loss.detach().cpu().numpy())
                hedging_error_train = np.append(hedging_error_train, hedging_error.detach().cpu().numpy())
                exercised = np.append(exercised, self.condition.detach().cpu().numpy())
                strat = np.append(strat, np.mean(strategy.detach().cpu().numpy()))
                maxAt = np.append(maxAt, np.max(A_t_tensor.detach().cpu().numpy()))
                maxBt = np.append(maxBt, np.max(B_t_tensor.detach().cpu().numpy()))

            if lr_schedule:
                self.scheduler.step()

            # print("DELTA_T_NEXT: " , self.delta_t_next)
            # print("STRATEGY: ", strategy.detach().numpy()[:, 10, -1])

            # Store the training loss after each epoch
            self.losses_epochs = np.append(self.losses_epochs, np.mean(losses))
            # Print stats
            if (epoch + 1) % 1 == 0:
                print("Time elapsed:", dt.datetime.now() - start)
                print("Epoch: %d, %s, Train Loss: %.3f" % (epoch + 1, self.loss_type, self.losses_epochs[epoch]))
                print("Proportion of exercise: ", np.mean(exercised))
                print("Strike: ", self.strike)
                if lr_schedule:
                    print("Learning rate: ", str(self.optimizer.param_groups[0]["lr"]))

            # Save the model if it's better
            # if self.losses_epochs[epoch] < best_loss:
            #     best_loss = self.losses_epochs[epoch]
            #     torch.save(self.model, "/home/a_eagu/Deep-Hedging-with-Market-Impact/" + self.name)

            # # Early stop after training on more epoch
            # if early_stop:
            #     break

            # # Early stopping criteria
            # if self.losses_epochs[epoch] > best_loss:
            #     worse_loss += 1
            #     if worse_loss == 2:
            #         early_stop = True

            epoch += 1

        torch.save(self.model, self.name)

        return all_losses, self.losses_epochs
    
    def testing(self, test_size, test_set):
        hedging_err_pred = []
        strategy_pred = []
        S_t_tensor_pred = []
        V_t_tensor_pred = []
        A_t_tensor_pred = []
        B_t_tensor_pred = []

        self.model.eval()

        for i in range(int(test_size/self.batch_size)):
            with torch.no_grad():
                
                if i % 100 == 0:
                    print("BATCH: " + str(i) + "/" + str(int(test_size/self.batch_size)))

                batch = test_set[i*self.batch_size:(i+1)*self.batch_size,:]
                hedging_error, strategy, S_t_tensor, V_t_tensor, A_t_tensor, B_t_tensor = self.simulate_batch(batch=batch)

                strategy_pred.append(strategy.detach().cpu().numpy())
                hedging_err_pred.append(hedging_error.detach().cpu().numpy())
                S_t_tensor_pred.append(S_t_tensor.detach().cpu().numpy())
                V_t_tensor_pred.append(V_t_tensor.detach().cpu().numpy())
                A_t_tensor_pred.append(A_t_tensor.detach().cpu().numpy())
                B_t_tensor_pred.append(B_t_tensor.detach().cpu().numpy())
        
        return np.concatenate(strategy_pred, axis=1), np.concatenate(hedging_err_pred), np.concatenate(S_t_tensor_pred, axis=1), np.concatenate(V_t_tensor_pred, axis=1), np.concatenate(A_t_tensor_pred, axis=1), np.concatenate(B_t_tensor_pred, axis=1)

    def point_predict(self, t, S_t, V_t, A_t, B_t, delta_t):
        S_t = torch.tensor([S_t])
        A_t = torch.tensor([A_t])
        B_t = torch.tensor([B_t])
        t = torch.tensor([t])
        delta_t = torch.tensor([delta_t])
        V_t = torch.tensor([V_t], dtype=torch.float32)

        # Processing stock price
        if self.prepro_stock == "log":
            S_t = torch.log(S_t)
        elif self.prepro_stock == "log-moneyness":
            S_t = torch.log(S_t/self.strike)

        input_t = torch.stack((self.dt * t, S_t, delta_t, V_t/self.V_0, A_t, B_t), dim=1).to(device=self.device)
        
        with torch.no_grad():
            delta_t_next = self.model(input_t)

        return delta_t_next[0, 0].item()