import datetime as dt
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Utils_general
import torch.optim.lr_scheduler as lr_scheduler

class GRU_multilayer_cell(nn.Module):
    
    def __init__(self, batch_size, input_size, hidden_size, num_layers, device, dropout=0.5):
        
        super().__init__()

        self.first_gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        self.gru_cells = nn.ModuleList([nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

        self.dropout = nn.Dropout(dropout)

        self.batch_size = batch_size
        self.nbs_units = hidden_size
        self.device = device
        self.num_layers = num_layers

    def forward(self, x, hs):
        
        hs[0] = self.first_gru_cell(x, hs[0])
        hs[0] = self.dropout(hs[0])
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                hs[i+1] = self.gru_cells[i](hs[i], hs[i+1])
                hs[i+1] = self.dropout(hs[i+1])
            x = self.linear(hs[i+1])
        else:
            x = self.linear(hs[0])
        return x, hs

class DeepAgent():
    
    def __init__(self, nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                 nbs_shares, lambdas, name='model'):
        
        self.nbs_point_traj = nbs_point_traj
        self.batch_size = batch_size
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
        self.N = self.nbs_point_traj - 1    #number of time-steps
        self.dt = self.T / self.N    # time_step size

        self.A_0 = 0    #initial value of persistence impact for the ask
        self.lambda_a = lambdas[0]    #persistence parameter for the ask
        self.B_0 = 0    #initial value of persistence impact for the bid
        self.lambda_b = lambdas[1]    #persistence parameter for the bid
        self.params_vect = params_vect
        self.strike = strike

        # Device the computations will take place on
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Model
        self.model = GRU_multilayer_cell(batch_size=self.batch_size, input_size=3, hidden_size=self.nbs_units, num_layers=self.nbs_layers, device=self.device, dropout=dropout).to(self.device)

        self.name = name
        print("Initial value of the portfolio: ", V_0)

    # Simulate a batch of paths and hedging errors
    def simulate_batch(self, test_path=None):

        hs = [torch.zeros(self.batch_size, self.nbs_units, device=self.device) for i in range(self.nbs_layers)]

        self.delta_t = torch.zeros(self.batch_size, device=self.device) #number of shares at each time step
        # Extract model parameters
        if self.stock_dyn == "BSM":
            self.mu, self.sigma = self.params_vect
        
        # Portfolio value prior to trading at each time-step
        # If long, you buy the option by borrowing V_0 from the bank
        if self.position_type == "long":
            V_t = -self.V_0 * torch.ones(self.batch_size, device=self.device)
        
        # If short, you recieve the premium that you put in the bank
        elif self.position_type == "short":
            V_t = self.V_0 * torch.ones(self.batch_size, device=self.device)

        # Unsqueeze to add a dimension at axis 0
        self.V_t_tensor = torch.unsqueeze(V_t, axis=0)
        self.S_t_tensor = torch.unsqueeze(self.S_0 * torch.ones(self.batch_size, device=self.device), axis=0)
        self.A_t_tensor = torch.unsqueeze(self.A_0 * torch.ones(self.batch_size, device=self.device), axis=0)
        self.B_t_tensor = torch.unsqueeze(self.B_0 * torch.ones(self.batch_size, device=self.device), axis=0)

        # Processing stock price
        if self.prepro_stock == "log":
            self.S_t = math.log(self.S_0) * torch.ones(self.batch_size, device=self.device)
        elif self.prepro_stock == "log-moneyness":
            self.S_t = math.log(self.S_0 / self.strike) * torch.ones(self.batch_size, device=self.device)
        elif self.prepro_stock == "none":
            self.S_t = self.S_0 * torch.ones(self.batch_size, device=self.device)

        A_t = self.A_0 * torch.ones(self.batch_size, device=self.device)
        B_t = self.B_0 * torch.ones(self.batch_size, device=self.device)

        for t in range(self.N):
            # Construct feature vector at the beginning of time t
            # input_t[i, :] = [S_t, delta_t, V_t, A_t, B_t]
            # S_t ad V_t are normalized
            # input_t.shape = [batch_size, 6]
            input_t = torch.stack((self.dt * t * torch.ones(self.batch_size, device=self.device), self.S_t, self.delta_t), dim=1)
            
            # un-normalize price
            self.S_t = self.inverse_processing(self.S_t)
            
            #This is only to output self.input_t_tensor out of function to make sure the input works
            if t == 0:
                self.input_t_tensor = torch.unsqueeze(input_t, dim=0)
            else:
                self.input_t_tensor = torch.cat((self.input_t_tensor, torch.unsqueeze(input_t, dim=0)), dim=0)

            # Output of the model
            self.delta_t_next, hs = self.model(input_t, hs)

            # Once the hedge is computed: 1) compile in self.strategy; 2) update M_t (cash reserve)
            if t == 0:
                self.strategy = torch.unsqueeze(self.delta_t_next, dim=0)
                diff_delta_t = self.delta_t_next[:, 0]
                cashflow = self.liquid_func(self.S_t, -diff_delta_t, A_t, B_t)
                self.M_t = V_t + cashflow   #time-t amount in the bank account (cash reserve)
            else:
                self.strategy = torch.cat((self.strategy, torch.unsqueeze(self.delta_t_next, dim=0)), dim=0)
                # Compute amount in cash reserve
                diff_delta_t = self.delta_t_next[:, 0] - self.strategy[t-1, :, 0]
                cashflow = self.liquid_func(self.S_t, -diff_delta_t, A_t, B_t)
                self.M_t = self.int_rate_bank(self.M_t) + cashflow  # time-t amount in cash reserve

            # Compute liquidity impact and persistence
            # For ask:
            if self.lambda_a == -1:
                A_t = torch.zeros(self.batch_size, device=self.device)
            else:
                impact_ask = torch.where(diff_delta_t > 0, diff_delta_t, 0.0)
                A_t = (A_t + impact_ask) * math.exp(-self.lambda_a * self.dt)
            # For bid:
            if self.lambda_b == -1:
                B_t = torch.zeros(self.batch_size, device=self.device)
            else:
                impact_bid = torch.where(diff_delta_t < 0, -diff_delta_t, 0.0)
                B_t = (B_t + impact_bid) * math.exp(-self.lambda_b * self.dt)

            # Update features for next time step (market impact persistence already updated)
            # Update stock price
            if test_path is not None:
                self.S_t = test_path[t+1]
            else:
                if self.stock_dyn == "BSM":
                    Z = torch.randn(self.batch_size, device=self.device)
                    self.S_t = self.S_t * torch.exp((self.mu - self.sigma ** 2 / 2) * self.dt + self.sigma * math.sqrt(self.dt) * Z)

            self.S_t_tensor = torch.cat((self.S_t_tensor, torch.unsqueeze(self.S_t, dim=0)), dim=0)
            self.delta_t = self.strategy[t, :, 0]

            # Liquidation portfolio value
            L_t = self.liquid_func(self.S_t, self.delta_t, A_t, B_t)    #Revenue from liquidating
            V_t = self.int_rate_bank(self.M_t) + L_t    # Portfolio value
            self.V_t_tensor = torch.cat((self.V_t_tensor, torch.unsqueeze(V_t, dim=0)), dim=0)
            # Store market persistence values
            self.A_t_tensor = torch.cat((self.A_t_tensor, torch.unsqueeze(A_t, dim=0)), dim=0)
            self.B_t_tensor = torch.cat((self.B_t_tensor, torch.unsqueeze(B_t, dim=0)), dim=0)

            # Processing stock price
            self.S_t_pre = self.S_t
            if self.prepro_stock == "log":
                self.S_t = torch.log(self.S_t_pre)
            elif self.prepro_stock == "log-moneyness":
                self.S_t = torch.log(self.S_t_pre/self.strike)

        # Compute hedging error at maturity
        # Check if worth it to execute or not
        # Currently only working for call options
        self.M_t = self.int_rate_bank(self.M_t)
        self.S_t = self.inverse_processing(self.S_t)

        # If call option: buyer executes iif profit selling > K 
        if self.position_type == "short":
            if self.option_type == "call":
                self.condition = torch.where(self.cost_selling(self.S_t, self.nbs_shares, B_t) >= self.nbs_shares * self.strike, 1, 0)
                self.hedging_gain = torch.where(self.cost_selling(self.S_t, self.nbs_shares, B_t) >= self.nbs_shares * self.strike, 
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t - self.nbs_shares, A_t, B_t) + self.nbs_shares * self.strike, 
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
            if self.option_type == "put":
                self.condition = torch.where(self.cost_buying(self.S_t, self.nbs_shares, B_t) <= self.nbs_shares * self.strike, 1, 0)
                self.hedging_gain = torch.where(self.cost_buying(self.S_t, self.nbs_shares, A_t) <= self.nbs_shares * self.strike,
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t + self.nbs_shares, A_t, B_t) - self.nbs_shares * self.strike,
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
        if self.position_type == "long":
            if self.option_type == "call":
                self.condition = torch.where(self.cost_selling(self.S_t, self.nbs_shares, B_t) >= self.nbs_shares * self.strike, 1, 0)
                self.hedging_gain = torch.where(self.cost_selling(self.S_t, self.nbs_shares, B_t) >= self.nbs_shares * self.strike,
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t + self.nbs_shares, A_t, B_t) - self.nbs_shares * self.strike,
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
            if self.option_type == "put":
                self.condition = torch.where(self.cost_buying(self.S_t, self.nbs_shares, B_t) <= self.nbs_shares * self.strike, 1, 0)
                self.hedging_gain = torch.where(self.cost_buying(self.S_t, self.nbs_shares, A_t) <= self.nbs_shares * self.strike,
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t - self.nbs_shares, A_t, B_t) + self.nbs_shares * self.strike,
                                                self.M_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
        self.hedging_error = -self.hedging_gain
        
        return self.hedging_error, self.strategy, self.S_t_tensor, self.V_t_tensor, self.A_t_tensor, self.B_t_tensor

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
    def loss(self, hedging_error):
        if self.loss_type == "RMSE":
            loss = torch.sqrt(torch.mean(torch.square(hedging_error)))
        elif self.loss_type == "RMSE per share":
            loss = torch.sqrt(torch.mean(torch.square(hedging_error))) / self.nbs_shares
        elif self.loss_type == "RSMSE":
            loss = torch.sqrt(torch.mean(torch.square(torch.where(hedging_error > 0, hedging_error, 0))))
        elif self.loss_type == "RSMSE per share":
            loss = torch.sqrt(torch.mean(torch.square(torch.where(hedging_error > 0, hedging_error, 0)))) / self.nbs_shares
        return loss

    def train(self, train_size, epochs, lr_schedule = True):
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
            # hs = self.model.init_hidden_state()
            # cs = self.model.init_cell_state()

            # mini batch training
            for i in range(int(train_size/self.batch_size)):
                if i % 100 == 0:
                    print("BATCH: " + str(i) + "/" + str(int(train_size/self.batch_size)))
                # Zero out gradients
                self.optimizer.zero_grad()

                # Simulate batch
                hedging_error, strategy, S_t_tensor, V_t_tensor, A_t_tensor, B_t_tensor = self.simulate_batch()

                # Compute and backprop loss
                loss = self.loss(hedging_error)
                loss.backward()

                # hs = [h.detach() for h in hs]
                # cs = [c.detach() for c in cs]
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
        
        torch.save(self.model, "/home/a_eagu/Deep-Hedging-with-Market-Impact/" + self.name)

        return all_losses, self.losses_epochs
    
    def test(self, test_size, test_set):
        hedging_err_pred = []
        strategy_pred = []
        S_t_tensor_pred = []
        V_t_tensor_pred = []
        A_t_tensor_pred = []
        B_t_tensor_pred = []

        self.model.eval()

        for i in range(int(test_size/self.batch_size)):
            with torch.no_grad():

                test_path = test_set[i]
                hedging_error, strategy, S_t_tensor, V_t_tensor, A_t_tensor, B_t_tensor = self.simulate_batch(test_path)

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
        V_t = torch.tensor([V_t])

        # Processing stock price
        if self.prepro_stock == "log":
            S_t = torch.log(S_t)
        elif self.prepro_stock == "log-moneyness":
            S_t = torch.log(S_t/self.strike)

        input_t = torch.stack((self.dt * t, S_t, delta_t, V_t/self.V_0, A_t, B_t), dim=1)

        with torch.no_grad():
            delta_t_next = self.model(input_t)

        return delta_t_next[0, 0].item()