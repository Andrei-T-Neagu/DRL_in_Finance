import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import yfinance as yf

class DCC_GARCH():
    """
        r_t = mu_t + a_t
        a_t = H_t^{1/2} z_t
        H_t = D_t R_t D_t
        
        where

        r_t:        n x 1 vector of log returns of n assets at time t.
        a_t:        n x 1 vector of mean-corrected returns of n assets at time t, i.e E[a_t] = 0,
                    Cov[a_t] = H_t.
        mu_t:       n x 1 vector of the expected value of the conditional r_t.
        H_t:        n x n matrix of conditional variances of a_t at time t.
        H_t^{1/2}:  Any n x n matrix at time t such that H_t is the conditional variance matrix of a_t.
                    H_t^{1/2} may be obtained by a Cholesky factorization of H_t
        D_t:        n x n, diagonal matrix of conditional standard deviations of a_t at time t.
        R_t:        n x n, conditional correlation matrix of a_t at time t.
        z_t:        n x 1 vector of iid errors such that E[z_t] = 0 and E[z_t z_t^T] = I.
    """
    def __init__(self, data, type="vanilla"):
        """
            Args:
            data: (N,T) numpy array of T timesteps of N asset log returns
        """
        self.type = type
        self.N = data.shape[0]                                                      # Number of assets
        self.T = data.shape[1]                                                      # NUmber of time steps
        self.data = data                                                            #(N, T) Dataset of log returns
        self.r_data = data                                                          #(N, T) Log returns of the data
        self.mu = np.mean(self.r_data, axis=1, keepdims=True)                       #(N, 1) Expected value of the log returns of the data
        self.a_data = self.r_data - np.mean(self.r_data, axis=1, keepdims=True)     #(N, T) Mean corrected returns of the dataset
        self.H = np.cov(self.a_data)                                                #(N, N) conditional variance matrix 
        self.H_factor = np.linalg.cholesky(self.H)                                  #(N, N) cholesky factorization of the covariance matrix
        self.D = np.diag(np.std(self.a_data, axis=1))                               #(N, N) diagonal matrix of the standard deviation
        self.R = np.corrcoef(self.a_data)                                           #(N, N) conditional correlation matrix 
        self.a_return = self.H_factor@np.random.randn(self.N, 1)                    #(N, 1) generated a_t
        self.r = self.mu + self.a_return                                            #(N, 1) generate r_t
        self.e = np.linalg.inv(self.D)@self.a_return                                #(N, 1) generated e_t
        self.e_data = np.linalg.inv(self.D)@self.a_data                             #(N, T) e_t from data
        self.Q_bar = self.e_data@self.e_data.T/self.T                               #(N, N) Q_bar using e_data                                          
        self.num_params_garch = 4 if self.type == "gjr" else 3
        if type == "gjr":
            self.params = np.concatenate((np.var(self.a_data, axis=1), np.zeros(self.N*2)+0.2, np.zeros(self.N)+0.1, np.zeros(2)+0.2))   # parameters [N*alpha0, N*alpha, N*beta, N*gamma, a, b]
        else:
            self.params = np.concatenate((np.var(self.a_data, axis=1), np.zeros(self.N*2+2)+0.2))   # parameters [N*alpha0, N*alpha, N*beta, a, b]
        
        # self.params = [8.26653925e-05, 4.26930333e-05, 2.15645253e-05, 3.47352036e-05, 2.17463059e-05, 2.77256555e-05, 
        #                4.88055909e-01, 4.53632078e-01, 4.55335336e-01, 4.44909987e-01, 4.50541641e-01, 4.60263855e-01,
        #                5.10702325e-01, 4.91397344e-01, 5.00709648e-01, 4.96429990e-01, 5.08129108e-01, 5.19125001e-01, 
                    #    2.01240559e-01, 3.96732870e-01]                              # GARCH "AAPL ^GDAXI ^IXIC ^GSPC ^DJI MCD"
        # self.params = [1.19523637e-05, 2.11605229e-06, 
        #                1.03456625e-01, 1.04024146e-01, 
        #                8.79042244e-01, 8.79606348e-01, 
        #                2.33099997e-02, 9.72480707e-01]                              # GARCH "AAPL ^GSPC"                               
        # self.params = [1.22778074e-05, 1.78956576e-06, 6.09289961e-02, 4.69337514e-03,
        #                8.76762304e-01, 9.10380382e-01, 9.20651668e-02, 1.33300686e-01, 
        #                2.69344690e-02, 9.67347998e-01]                                # GJR-GARCH "AAPL ^GSPC"
        self.nll_losses = []                                                          # negative log likelihood array used to store training losses
        
        def constraint_ab(x):
            """DCC constraint a + b < 1"""
            return 1.0 - x[-2] - x[-1]
        
        if self.type == "gjr":
            def constraint_alpha_beta_gamma(x):
                """GARCH constraint alpha_n + beta_n < 1"""
                return 1.0 - x[self.N:2*self.N] - x[2*self.N:3*self.N] - x[3*self.N:4*self.N]/2.0
            
            self.constraints = [{'type':'ineq', 'fun':constraint_ab},
                                {'type':'ineq', 'fun':constraint_alpha_beta_gamma}]
        else:
            def constraint_alpha_beta(x):
                """GARCH constraint alpha_n + beta_n < 1"""
                return 1.0 - x[self.N:2*self.N] - x[2*self.N:3*self.N]
            
            self.constraints = [{'type':'ineq', 'fun':constraint_ab},
                                {'type':'ineq', 'fun':constraint_alpha_beta}]
        
        self.bounds = Bounds([np.finfo(float).eps for i in range(self.N*self.num_params_garch+2)],        # Bounds (0,inf) for alpha_0, alpha, beta, gamma(if gjr), a, b 
                             [np.inf for i in range(self.N*self.num_params_garch+2)], 
                             keep_feasible=[True for i in range(self.N*self.num_params_garch+2)])

    def D_t(self, params, t=0, train=False):
        """
            h_{n,t} = alpha0_{n} + alpha_{n} a_{n, t-1}^2 + beta_{n} h_{n, t-1}
        """
        alpha0 = np.expand_dims(params[0:self.N],axis=1)
        alpha = np.expand_dims(params[self.N:2*self.N],axis=1)
        beta = np.expand_dims(params[2*self.N:3*self.N], axis=1)
        if self.type == "gjr":
            gamma = np.expand_dims(params[3*self.N:4*self.N], axis=1)
            I = np.where(self.a_return > 0, 0.0, 1.0)
            alpha = alpha + gamma*I

        h_vector = np.expand_dims(np.diag(self.D)**2, axis=1)
        new_h_vector = alpha0 + alpha * self.a_return**2 + beta * h_vector
        self.D = np.diag(np.sqrt(new_h_vector.flatten()))
        
        return self.D
    
    def R_t(self, params, t=0, train=False):
        """
            e_t = D_t^{-1} a_t \\sim N(0, R_t)
            R_t = Q_t^{*-1} Q_t Q_t^{*-1}
            Q_t = (1 - a - b) Q_bar + a e_{t-1} e_{t-1}^T + b Q_{t-1}
        """
        a = params[-2]
        b = params[-1]

        self.Q = (1 - a - b) * self.Q_bar + a * self.e@self.e.T + b * self.Q
        Q_diag_inv = np.linalg.inv(np.diag(np.sqrt(np.diag(self.Q))))
        self.R = Q_diag_inv@self.Q@Q_diag_inv

        return self.R

    def r_t(self, t=0, train=False):
        """
        r_t = mu + a_t
        a_t = H_factor * z_t
        H_t = D_t R_t D_t
        """
        self.H = self.D@self.R@self.D
        self.H_factor = np.linalg.cholesky(self.H)
        self.a_return = self.H_factor@np.random.randn(self.N, 1)
        self.r = self.mu + self.a_return
        self.e = np.linalg.inv(self.D)@self.a_return
        return self.r
    
    def neg_log_likelihood(self, params):
        """
        Compute negative log likelihhod
        1/2 sum^T_{t=1} [n * ln(2*pi) + 2 * ln(det(D_t)) + ln(det(R_t)) + a_t^T D_t^{-1} R_t^{-1} D_t^{-1} a_t]
        """
        # print("params: ", params)
        train = True
        log_likelihood = 0
        self.D = np.diag(np.std(self.a_data, axis=1))
        self.R = np.corrcoef(self.a_data)
        self.Q = self.Q_bar
        for t in range(self.T):
            self.a_return = self.a_data[:, t:t+1]
            current_log_likelihood = self.N * np.log(2*np.pi) + 2*np.log(np.linalg.det(self.D)) + np.log(np.linalg.det(self.R)) + self.a_return.T@np.linalg.inv(self.D)@np.linalg.inv(self.R)@np.linalg.inv(self.D)@self.a_return
            log_likelihood += current_log_likelihood
            self.e = np.linalg.inv(self.D)@self.a_return
            self.D_t(params=params, t=t, train=train)
            self.R_t(params=params, t=t, train=train)
        nll = log_likelihood.flatten()[0]/2.0
        # print("nll: ", nll)
        # print()
        self.nll_losses.append(nll)
        return nll
    
    def train(self):
        """Training loop"""
        params = self.params
        results = minimize(fun=self.neg_log_likelihood, x0=params, method="trust-constr",
                           options={'verbose': 3, 'maxiter': 2000, 'xtol': 1e-4}, constraints=self.constraints, bounds=self.bounds)
        self.params = results.x
        return results.x

    def generate(self, S_0, num_points):
        """Loop used to generate paths using the estimated parameters"""
        prices = np.ones((self.N, num_points)) * S_0                                # (N, num_points) Paths of length num_points of the N assets
        self.D = np.diag(np.std(self.a_data, axis=1))
        self.R = np.corrcoef(self.a_data)
        self.Q = self.Q_bar

        for t in range(num_points):
            r_t = self.r_t()
            prices[:, t+1:t+2] = prices[:, t:t+1]*np.exp(r_t)
            self.D_t(params=self.params)
            self.R_t(params=self.params)
        return prices
    
"""
^GDAXI: DAX, 
AAPL: APPLE,
^IXIC: NASDAQ
^GSPC: S AND P 500
"""
stocks = "AAPL ^GDAXI ^IXIC ^GSPC ^DJI MCD"
# stocks = "AAPL ^GSPC"

data = yf.download(stocks, start="2000-09-10", end="2024-08-20", interval="1d")
log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
log_returns_np = log_returns.to_numpy().T

model = DCC_GARCH(log_returns_np, type="gjr")

params = model.train()
print(params)

plt.figure(figsize=(12,6))
plt.plot(model.nll_losses)
plt.xlabel("Iterations")
plt.ylabel("Negative Log Likelihood")
plt.savefig("dcc_nll_losses.png")
plt.close()

plt.figure(figsize=(12,6))
x = model.generate(100, 252*5)
plt.plot(x.T)
plt.xlabel("Timesteps")
plt.ylabel("Prices")
plt.legend(list(stocks.split(" ")))
plt.savefig("dcc_garch_test.png")
plt.close()