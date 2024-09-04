import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import yfinance as yf
import pickle
import GARCH

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
    def __init__(self, stocks, start="2000-01-01", end="2024-08-20", type="vanilla"):
        """
            Args:
            data: (N,T) numpy array of T timesteps of N asset log returns
        """
        market_data = yf.download(stocks, start=start, end=end, interval="1d")
        log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
        self.data = log_returns.to_numpy().T                                        #(N, T) Dataset of log returns
        print("Number of time steps: ", self.data.shape[1])
        self.type = type
        self.N = self.data.shape[0]                                                 # Number of assets
        self.T = self.data.shape[1]                                                 # Number of time steps                                                                                    
        self.r_data = self.data                                                     #(N, T) Log returns of the data
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
            self.params = np.concatenate((np.var(self.a_data, axis=1), np.zeros(self.N*2)+0.2, np.zeros(2)+0.2))   # parameters [N*alpha0, N*alpha, N*beta, a, b]
        
        self.nll_losses = []                                                         # negative log likelihood array used to store training losses
        
        def constraint_ab(x):
            """DCC constraint a + b < 1"""
            return 1.0 - x[-2] - x[-1]
        
        if self.type == "gjr":
            def constraint_alpha_beta_gamma(x):
                """GARCH constraint alpha_n + beta_n < 1"""
                return 1.0 - x[self.N:2*self.N] - x[2*self.N:3*self.N] - x[3*self.N:4*self.N]/2.0
            
            self.garch_constraints = [{'type':'ineq', 'fun':constraint_alpha_beta_gamma}]
        else:
            def constraint_alpha_beta(x):
                """GARCH constraint alpha_n + beta_n < 1"""
                return 1.0 - x[self.N:2*self.N] - x[2*self.N:3*self.N]
            
            self.garch_constraints = [{'type':'ineq', 'fun':constraint_alpha_beta}]
        
        self.dcc_constraints = [{'type':'ineq', 'fun':constraint_ab}]

        self.garch_bounds = Bounds([np.finfo(float).eps for i in range(self.N*self.num_params_garch)],        # Bounds (0,inf) for alpha_0, alpha, beta, gamma(if gjr)
                             [np.inf for i in range(self.N*self.num_params_garch)], 
                             keep_feasible=[True for i in range(self.N*self.num_params_garch)])

        self.dcc_bounds = Bounds([np.finfo(float).eps for i in range(2)],        # Bounds (0,inf) for a, b 
                             [np.inf for i in range(2)], 
                             keep_feasible=[True for i in range(2)])

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
    
    def D_t_batch(self, params, batch_size, t=0, train=False):
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

        h_vector = np.expand_dims(np.diagonal(self.D, axis1=-2, axis2=-1)**2, axis=2)
        new_h_vector = alpha0 + alpha * self.a_return**2 + beta * h_vector
        new_h_vector = np.reshape(new_h_vector, (batch_size, self.N))
        mydiag=np.vectorize(np.diag, signature='(n)->(n,n)')
        self.D = mydiag(np.sqrt(new_h_vector))
        
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
    
    def R_t_batch(self, params, batch_size, t=0, train=False):
        """
            e_t = D_t^{-1} a_t \\sim N(0, R_t)
            R_t = Q_t^{*-1} Q_t Q_t^{*-1}
            Q_t = (1 - a - b) Q_bar + a e_{t-1} e_{t-1}^T + b Q_{t-1}
        """
        a = params[-2]
        b = params[-1]

        self.Q = (1 - a - b) * self.Q_bar + a * self.e@np.transpose(self.e, axes=(0,2,1)) + b * self.Q
        mydiag=np.vectorize(np.diag, signature='(n)->(n,n)')
        Q_diag_inv = np.linalg.inv(mydiag(np.sqrt(np.diagonal(self.Q, axis1=-2, axis2=-1))))
        self.R = Q_diag_inv@self.Q@Q_diag_inv
        return self.R

    def r_t(self, batch_size, t=0, train=False):
        """
        r_t = mu + a_t
        a_t = H_factor * z_t
        H_t = D_t R_t D_t
        """
        self.H = self.D@self.R@self.D
        self.H_factor = np.linalg.cholesky(self.H)
        self.a_return = self.H_factor@np.random.randn(batch_size, self.N, 1)
        self.r = self.mu + self.a_return
        self.e = np.linalg.inv(self.D)@self.a_return

        return self.r
    
    def garch_neg_log_likelihood(self, params):
        """
        Compute negative log likelihhod
        1/2 sum^T_{t=1} [n * ln(2*pi) + 2 * ln(det(D_t)) + ln(det(R_t)) + a_t^T D_t^{-1} R_t^{-1} D_t^{-1} a_t]
        """
        # print("params: ", params)
        train = True
        log_likelihood = 0
        self.D = np.diag(np.std(self.a_data, axis=1))
        # self.R = np.corrcoef(self.a_data)
        Id = np.identity(self.N)
        self.Q = self.Q_bar
        for t in range(self.T):
            self.a_return = self.a_data[:, t:t+1]
            current_log_likelihood = self.N * np.log(2*np.pi) + 2*np.log(np.linalg.det(self.D)) + self.a_return.T@np.linalg.inv(self.D)@Id@np.linalg.inv(self.D)@self.a_return
            log_likelihood += current_log_likelihood
            # self.e = np.linalg.inv(self.D)@self.a_return
            self.D_t(params=params, t=t, train=train)
            # self.R_t(params=params, t=t, train=train)
        nll = log_likelihood.flatten()[0]/2.0
        # print("nll: ", nll)
        # print()
        self.nll_losses.append(nll)
        return nll
    
    def dcc_neg_log_likelihood(self, params):
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
            current_log_likelihood = np.log(np.linalg.det(self.R)) + self.a_return.T@np.linalg.inv(self.D)@np.linalg.inv(self.R)@np.linalg.inv(self.D)@self.a_return
            log_likelihood += current_log_likelihood
            self.e = np.linalg.inv(self.D)@self.a_return
            self.D_t(params=self.params, t=t, train=train)
            self.R_t(params=params, t=t, train=train)
        nll = log_likelihood.flatten()[0]/2.0
        # print("nll: ", nll)
        # print()
        self.nll_losses.append(nll)
        return nll
    
    def train(self, save_params = False):
        """Training loop"""
        garch_params = self.params[0:-2]
        garch_results = minimize(fun=self.garch_neg_log_likelihood, x0=garch_params, method="trust-constr",
                           options={'verbose': 3, 'maxiter': 2000, 'xtol': 1e-4}, constraints=self.garch_constraints, bounds=self.garch_bounds)
        self.params[0:-2] = garch_results.x
        
        dcc_params = self.params[-2:]
        dcc_results = minimize(fun=self.dcc_neg_log_likelihood, x0=dcc_params, method="trust-constr",
                           options={'verbose': 3, 'maxiter': 2000, 'xtol': 1e-8}, constraints=self.dcc_constraints, bounds=self.dcc_bounds)
        self.params[-2:] = dcc_results.x
        
        if save_params:
            with open('dcc_garch_parameters.pickle', 'wb') as parameter_file:
                pickle.dump(self.params, parameter_file)

        return self.params

    def train_R(self, save_params=False):
        parameters = []

        for i in range(len(list(stocks.split(" ")))):
            log_returns_i = self.data[i]
            garch_model = GARCH.GARCH(dcc=True, data=log_returns_i, type=self.type)
            params_i = garch_model.train()
            parameters.append(params_i)

        parameters = np.array(parameters)
        parameters = np.concatenate((parameters.T.flatten(), np.zeros(2)+0.2))

        self.params = parameters
        dcc_params = self.params[-2:]
        dcc_results = minimize(fun=self.dcc_neg_log_likelihood, x0=dcc_params, method="trust-constr",
                           options={'verbose': 3, 'maxiter': 2000, 'xtol': 1e-8}, constraints=self.dcc_constraints, bounds=self.dcc_bounds)
        self.params[-2:] = dcc_results.x

        if save_params:
            with open('dcc_garch_parameters.pickle', 'wb') as parameter_file:
                pickle.dump(self.params, parameter_file)

        return self.params
    
    def generate(self, S_0, batch_size, num_points, load_params=False):
        """Loop used to generate paths using the estimated parameters"""
        if load_params:
            with open('dcc_garch_parameters.pickle', 'rb') as parameter_file:
                self.params = pickle.load(parameter_file)
        
        prices = np.ones((batch_size, self.N, num_points)) * S_0                                # (N, num_points) Paths of length num_points of the N assets
        self.D = np.tile(np.diag(np.std(self.a_data, axis=1)),(batch_size,1,1))
        self.R = np.tile(np.corrcoef(self.a_data),(batch_size,1,1))
        self.Q = np.tile(self.Q_bar,(batch_size,1,1))

        for t in range(num_points):
            print("timestep: ", t)
            r_t = self.r_t(batch_size)
            prices[:, :, t+1:t+2] = prices[:, :, t:t+1]*np.exp(r_t)
            self.D_t_batch(params=self.params, batch_size=batch_size)
            self.R_t_batch(params=self.params, batch_size=batch_size)
        return prices

"""
^GDAXI: DAX (Germany)
^IXIC: NASDAQ
^GSPC: S AND P 500
^DJI: Dow Jones Industrial Average
^RUT: Russell 2000
^N225: Nikkei 225 (Japan)
^HSI: Hang Seng Index (Hong Kong)
^NYA: NYSE Composite
^FCHI: CAC 40 (France)
MCD: McDonald's
AAPL: Apple
"""

stocks = "^GDAXI ^IXIC ^GSPC ^DJI ^RUT ^N225 ^HSI ^NYA ^FCHI"
# stocks = "AAPL ^GSPC"
garch_type = "gjr"

dcc_garch_model = DCC_GARCH(stocks=stocks, type=garch_type)

params = dcc_garch_model.train_R(save_params=True)
print(params)

plt.figure(figsize=(12,6))
plt.plot(dcc_garch_model.nll_losses)
plt.xlabel("Function Evaluations")
plt.ylabel("Negative Log Likelihood")
plt.savefig("dcc_nll_losses.png")
plt.close()

num_points=252*5

plt.figure(figsize=(12,6))
x = dcc_garch_model.generate(S_0=100, batch_size=2**20, num_points=252*5, load_params=True)
plt.plot(x[0].T)
plt.xlabel("Timesteps")
plt.ylabel("Prices")
plt.legend(list(stocks.split(" ")))
plt.savefig("dcc_garch_test.png")
plt.close()