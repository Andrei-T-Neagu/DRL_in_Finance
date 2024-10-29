import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pickle
import yfinance as yf

class GARCH():
    """
        r_t = mu_t + a_t
        a_t = h_t^{1/2} z_t
        h_t = alpha_0 + alpha * a_{t-1}^2 + beta * h_{t-1}
        
        where

        r_t:        log return of asset at time t.
        a_t:        mean-corrected returns of asset at time t.
        mu_t:       expected value of the conditional r_t.
        h_t:        conditional variance of a_t at time t.
        z_t:        iid errors such that E[z_t] = 0 and E[z_t z_t] = 1.
    """
    def __init__(self, dcc=False, data=None, stock=None, start="2000-01-01", end="2024-08-20", interval="1d", type = "vanilla"):
        """
            Args:   
            dcc:        True if used in the DCC_GARCH classes
            data:       (N,T) T stock prices of the N assets. Only used in the DCC_GARCH classes
            stock:      stock ticker to be used to download the yfinance data
            start:      "YYYY-MM-DD" used by yfinance as the start date to download stock data
            end:        "YYYY-MM-DD" used by yfinance as the end date to download stock data
            interval:   Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
            type:       "vanilla" or "gjr"
        """
        if dcc:
            self.data = data
        else:
            market_data = yf.download(stock, start=start, end=end, interval=interval)
            log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
            self.data = log_returns.to_numpy().T*100
        self.type = type                                                                                # Garch type
        self.T = self.data.shape[0]                                                                     # Number of time steps in the data
        self.r_data = self.data                                                                         #(T,) log returns of the asset
        self.mu = np.mean(self.r_data)                                                                  # expected value of r_t
        self.a_data = self.r_data - np.mean(self.r_data)                                                #(T,) mean-corrected returns of asset
        self.h = np.var(self.a_data)                                                                    # conditional variance of a_t
        self.a_return = np.sqrt(self.h) * np.random.randn(1)                                            # generated a_t
        self.num_params_garch = 4 if self.type == "gjr" or self.type == "e" else 3                      # number of parameters for the garch or gjr-garch model
        self.params = np.concatenate(([np.var(self.a_data)],np.zeros(self.num_params_garch-1)+0.3))     #(3,) or (4,) Set of parameters alpha_0, alpha, beta, gamma(if gjr)
        self.nll_losses = []                                                                            # Array to store nll losses during training

        if self.type == "vanilla":
            def constraint_alpha_beta(x):
                """constraint that alpha + beta < 1"""
                return 1.0 - x[1] - x[2]
        
            self.constraints = [{'type':'ineq', 'fun':constraint_alpha_beta}]

            self.bounds = Bounds([np.finfo(float).eps for i in range(self.num_params_garch)],               #(0,inf) bounds for alpha_0, alpha, beta, (gamma if gjr)
                                [1.0 for i in range(self.num_params_garch)], 
                                keep_feasible=[True for i in range(self.num_params_garch)])
        elif self.type == "gjr":
            def constraint_alpha_beta_gamma(x):
                """constraint that alpha + beta + gamma/2 < 1"""
                return 1.0 - x[1] - x[2] - x[3]/2.0
        
            self.constraints = [{'type':'ineq', 'fun':constraint_alpha_beta_gamma}]

            self.bounds = Bounds([np.finfo(float).eps for i in range(self.num_params_garch)],               #(0,inf) bounds for alpha_0, alpha, beta, (gamma if gjr)
                                [1.0 for i in range(self.num_params_garch)], 
                                keep_feasible=[True for i in range(self.num_params_garch)])
        elif self.type == "e":
            self.constraints = None
            self.bounds = None

    def h_t(self, params, t=0, train=False):
        """Updates h_t = alpha_0 + alpha * a_{t-1}^2 + beta * h_{t-1}"""
        alpha_0 = params[0]
        alpha = params[1]
        beta = params[2]
        if self.type == "e":
            gamma = params[3]

            self.h = np.exp(alpha_0 + alpha * (np.absolute(self.a_return/np.sqrt(self.h)) - np.sqrt(2.0/np.pi)) + gamma * (self.a_return/np.sqrt(self.h)) + beta * np.log(self.h))
        elif self.type == "gjr":
            gamma = params[3]
            I = np.where(self.a_return > 0, 0.0, 1.0)
            alpha = alpha + gamma*I
            
            self.h = alpha_0 + alpha * self.a_return**2 + beta * self.h
        elif self.type == "vanilla":
            self.h = alpha_0 + alpha * self.a_return**2 + beta * self.h

    def neg_log_likelihood(self, params):
        """
        Computes the negative log likelihood
        = 1/2 sum^T_{t=1}[ln(2*pi) + ln(h_t) + a_t^2/h_t]
        """
        train = True
        log_likelihood = 0
        self.h = np.var(self.a_data)
        for t in range(self.T):
            self.a_return = self.a_data[t]
            current_log_likelihood = np.log(2*np.pi) + np.log(self.h) + self.a_return**2/self.h
            log_likelihood += current_log_likelihood
            self.h_t(params=params, t=t, train=train)
        nll = log_likelihood/2.0
        self.nll_losses.append(nll)

        return nll
    
    def train(self, save_params=False):
        """Training loop"""
        params = self.params
        results = minimize(fun=self.neg_log_likelihood, x0=params, method="slsqp",
                           options={'disp': True, 'maxiter': 1000}, constraints=self.constraints, bounds=self.bounds)
        self.params = results.x
        
        # if self.type == "vanilla":
        #     print("GARCH estimated unconditional variance: ", self.params[0]/(1-self.params[1]-self.params[2]))
        # elif self.type == "gjr":
        #     print("GJR-GARCH estimated unconditional variance: ", self.params[0]/(1-self.params[1]-self.params[2]-self.params[3]/2))
        # print("Market data unconditional variance: ", np.var(self.r_data))
        
        if save_params:
            with open('garch_parameters.pickle', 'wb') as parameter_file:
                pickle.dump(self.params, parameter_file)
        
        return results.x

    def generate(self, S_0, batch_size, num_points, load_params=False):
        """Loop used to generate paths using the estimated parameters, where the batch size is specified"""
        if load_params:
            with open('garch_parameters.pickle', 'rb') as parameter_file:
                self.params = pickle.load(parameter_file)
        
        prices = np.ones((batch_size, num_points)) * S_0
        self.h = np.tile(np.var(self.a_data),(batch_size,1))
        
        for t in range(num_points):
            # if t % 100 == 0:
            #     print("timestep: " + str(t) + "/" + str(num_points))
            self.a_return = np.sqrt(self.h) * np.random.randn(batch_size,1)
            r_t = self.mu + self.a_return
            prices[:, t+1:t+2] = prices[:, t:t+1]*np.exp(r_t/100)
            self.h_t(params=self.params)
        
        return prices

    def print_params(self):
        """Function used to print the parameters of the model"""
        if self.type == "gjr" or self.type == "e":
            print("alpha_0: " + str(self.params[0]) + ", alpha: " + str(self.params[1]) + ", beta: " + str(self.params[2]) + ", gamma: " + str(self.params[3]))
        else:
            print("alpha_0: " + str(self.params[0]) + ", alpha: " + str(self.params[1]) + ", beta: " + str(self.params[2]))