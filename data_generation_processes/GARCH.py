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
    def __init__(self, dcc=False, data=None, stock=None, start="2000-01-01", end="2024-08-20", type = "vanilla"):
        """
            Args:
            data: (N,T) numpy array of T timesteps asset log returns
            type: "vanilla" or "gjr"
        """
        if dcc:
            self.data = data
        else:
            market_data = yf.download(stock, start=start, end=end, interval="1d")
            log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
            self.data = log_returns.to_numpy().T
        self.type = type
        self.T = self.data.shape[0]                                                 # Number of time steps in the data
        self.r_data = self.data                                                     #(T,) log returns of the asset
        self.mu = np.mean(self.r_data)                                              # expected value of r_t
        self.a_data = self.r_data - np.mean(self.r_data)                            #(T,) mean-corrected returns of asset
        self.h = np.var(self.a_data)                                                # conditional variance of a_t
        self.a_return = self.h * np.random.randn(1)                                 # generated a_t
        self.num_params_garch = 4 if self.type == "gjr" else 3                      # number of parameters for the garch or gjr-garch model
        self.params = np.concatenate(([np.var(self.a_data)],np.zeros(self.num_params_garch-1)+0.4))   #(3,) or (4,) Set of parameters alpha_0, alpha, beta, gamma(if gjr)
        self.nll_losses = []                                                        # To store nll losses during training

        if self.type == "vanilla":
            def constraint_alpha_beta(x):
                """constraint that alpha + beta < 1"""
                return 1.0 - x[1] - x[2]
        
            self.constraints = [{'type':'ineq', 'fun':constraint_alpha_beta}]

        else:
            def constraint_alpha_beta_gamma(x):
                """constraint that alpha + beta + gamma/2 < 1"""
                return 1.0 - x[1] - x[2] - x[3]/2.0
        
            self.constraints = [{'type':'ineq', 'fun':constraint_alpha_beta_gamma}]

        self.bounds = Bounds([np.finfo(float).eps for i in range(self.num_params_garch)],               #(0,inf) bounds for alpha_0, alpha, beta, (gamma if gjr)
                            [np.inf for i in range(self.num_params_garch)], 
                            keep_feasible=[True for i in range(self.num_params_garch)])

    def h_t(self, params, t=0, train=False):
        """Updates h_t = alpha_0 + alpha * a_{t-1}^2 + beta * h_{t-1}"""
        alpha_0 = params[0]
        alpha = params[1]
        beta = params[2]
        if self.type == "gjr":
            gamma = params[3]
            I = np.where(self.a_return > 0, 0.0, 1.0)
            alpha = alpha + gamma*I
            
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
        results = minimize(fun=self.neg_log_likelihood, x0=params, method="trust-constr",
                        options={'verbose': 3, 'disp': True, 'maxiter': 1000}, constraints=self.constraints, bounds=self.bounds)
        self.params = results.x
        
        if self.type == "vanilla":
            print("GARCH estimated unconditional variance: ", self.params[0]/(1-self.params[1]-self.params[2]))
        else:
            print("GJR-GARCH estimated unconditional variance: ", self.params[0]/(1-self.params[1]-self.params[2]-self.params[3]/2))
        print("Market data unconditional variance: ", np.var(self.a_data))
        
        if save_params:
            with open('garch_parameters.pickle', 'wb') as parameter_file:
                pickle.dump(self.params, parameter_file)
        
        return results.x

    def generate(self, S_0, num_points, load_params=False):
        """Loop used to generate paths using the estimated parameters"""
        if load_params:
            with open('garch_parameters.pickle', 'rb') as parameter_file:
                self.params = pickle.load(parameter_file)
        
        prices = np.ones(num_points) * S_0
        self.h = np.var(self.a_data)
        for t in range(num_points):
            self.a_return = np.sqrt(self.h) * np.random.randn(1)
            r_t = self.mu + self.a_return
            prices[t+1:t+2] = prices[t:t+1]*np.exp(r_t)
            self.h_t(params=self.params)
        
        return prices



stock = "AAPL"

model = GARCH(stock=stock, type="gjr")

# Training
# params = model.train(save_params=True)
# print("parameters: ", params)

# plt.figure(figsize=(12,6))
# plt.plot(model.nll_losses)
# plt.xlabel("Function Evaluations")
# plt.ylabel("Negative Log Likelihood")
# plt.savefig("garch_nll_losses.png")
# plt.close()

# Generating
x = model.generate(100, 252*5, load_params=True)
print(model.params)
plt.figure(figsize=(12,6))
plt.plot(x.T)
plt.xlabel("Timesteps")
plt.ylabel("Prices")
plt.legend([stock])
plt.savefig("garch_test.png")
plt.close()