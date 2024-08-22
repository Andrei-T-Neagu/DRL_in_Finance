import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    def __init__(self, data):
        """
            Args:
            data: (N,T) numpy array of T timesteps of N asset log returns
        """
        self.N = data.shape[0]
        self.T = data.shape[1]
        self.data = data
        self.r_data = data                                                          #(N, T)
        self.mu = np.mean(self.r_data, axis=1, keepdims=True)                       #(N, 1) 
        self.a_data = self.r_data - np.mean(self.r_data, axis=1, keepdims=True)     #(N, T)
        self.H = np.cov(self.a_data)                                                #(N, N)
        self.H_factor = np.linalg.cholesky(self.H)                                  #(N, N)
        self.D = np.diag(np.std(self.a_data, axis=1))                               #(N, N)
        self.R = np.corrcoef(self.a_data)                                           #(N, N)
        self.a_return = self.H_factor@np.random.randn(self.N, 1)                    #(N, 1)
        self.r = self.mu + self.a_return                                            #(N, 1)
        self.e = np.linalg.inv(self.D)@self.a_return                                #(N, 1)
        self.e_data = np.linalg.inv(self.D)@self.a_data                             #(N, T)
        self.Q_bar = self.e_data@self.e_data.T/self.T                               #(N, N)                                           
        self.Q = self.Q_bar                                                         #(N, N)
        # self.params = np.zeros(self.N*3+2)+0.05                                     #[N*alpha0, N*alpha, N*beta, a, b]
        # self.alpha0 = np.std(self.a_data, axis=1)
        # self.params = np.append(self.alpha0, np.zeros(self.N*2+2)+0.005)             #[N*alpha0, N*alpha, N*beta, a, b]
        # self.params = [8.31405269e-05, 4.71951483e-05, 8.88408298e-02, 1.05864782e-01, 2.24089113e-06, 1.00361059e-04, 2.45603818e-02, 1.47373622e-05]
        # self.params = [8.31405269e-05, 4.71951483e-05, 8.88408298e-02, 1.05864782e-01, 2.24089113e-06, 1.00361059e-04, 2.45603818e-02, 1.47373622e-05]
        # self.params = [7.12056487e-07, 2.78392987e-07, 2.35193830e-01, 2.22333143e-01, 4.24217714e-06, 1.80558148e-04, 9.97035213e-01, 8.78583222e-05]          #nll:  [[-50473.23009387]]
        self.params = np.zeros(self.N*3+2)+0.25                                     #[N*alpha0, N*alpha, N*beta, a, b]
        self.nll_losses = []

        def constraint_ab(x):
            return 1.0 - x[-2] - x[-1]

        def constraint_a(x):
            return x[-2]

        def constraint_b(x):
            return x[-1]

        def constraint_alpha_beta1(x):
            return 1.0 - x[self.N] - x[2*self.N]

        def constraint_alpha01(x):
            return x[0]
        
        def constraint_alpha1(x):
            return x[self.N]

        def constraint_beta1(x):
            return x[2*self.N]
        
        def constraint_alpha_beta2(x):
            return 1.0 - x[self.N+1] - x[2*self.N+1]

        def constraint_alpha02(x):
            return x[1]
        
        def constraint_alpha2(x):
            return x[self.N+1]

        def constraint_beta2(x):
            return x[2*self.N+1]

        def constraint_alpha_beta(x):
            return 1.0 - x[self.N:2*self.N] - x[2*self.N:3*self.N]

        def constraint_alpha0(x):
            return x[0:self.N]
        
        def constraint_alpha(x):
            return x[self.N:2*self.N]

        def constraint_beta(x):
            return x[2*self.N:3*self.N]

        #[alpha01, alpha02, alpha1, alpha2, beta1, beta2, a, b]
        # self.constraints = [{'type':'ineq', 'fun':constraint_ab},
        #                     {'type':'ineq', 'fun':constraint_a},
        #                     {'type':'ineq', 'fun':constraint_b},
        #                     {'type':'ineq', 'fun':constraint_alpha_beta1},
        #                     {'type':'ineq', 'fun':constraint_alpha01},
        #                     {'type':'ineq', 'fun':constraint_alpha1},
        #                     {'type':'ineq', 'fun':constraint_beta1},
        #                     {'type':'ineq', 'fun':constraint_alpha_beta2},
        #                     {'type':'ineq', 'fun':constraint_alpha02},
        #                     {'type':'ineq', 'fun':constraint_alpha2},
        #                     {'type':'ineq', 'fun':constraint_beta2}]
        
        self.constraints = [{'type':'ineq', 'fun':constraint_ab},
                            {'type':'ineq', 'fun':constraint_a},
                            {'type':'ineq', 'fun':constraint_b},
                            {'type':'ineq', 'fun':constraint_alpha_beta},
                            {'type':'ineq', 'fun':constraint_alpha0},
                            {'type':'ineq', 'fun':constraint_alpha},
                            {'type':'ineq', 'fun':constraint_beta}]

    def D_t(self, params, t=0, train=False):
        """
            h_{n,t} = alpha0_{n} + alpha_{n} a_{n, t-1}^2 + beta_{n} h_{n, t-1}
        """
        alpha0 = np.expand_dims(params[0:self.N],axis=1)
        alpha = np.expand_dims(params[self.N:2*self.N],axis=1)
        beta = np.expand_dims(params[2*self.N:3*self.N], axis=1)
        
        h_vector = np.expand_dims(np.diag(self.D)**2,axis=1)
        if train:
            self.a_return = self.a_data[:, t:t+1]
        new_h_vector = alpha0 + alpha * self.a_return**2 + beta * h_vector
        if np.any(new_h_vector<0):
            print(new_h_vector)
            print(alpha0)
            print(alpha)
            print(beta)
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

        if train:
            self.e = self.e_data[:, t:t+1]
        self.Q = (1 - a - b) * self.Q_bar + a * self.e@self.e.T + b * self.Q
        Q_diag_inv = np.linalg.inv(np.diag(np.sqrt(np.diag(self.Q))))
        self.R = Q_diag_inv@self.Q@Q_diag_inv
        
        return self.R

    def r_t(self, t=0, train=False):
        self.H = self.D@self.R@self.D
        self.H_factor = np.linalg.cholesky(self.H)
        self.a_return = self.H_factor@np.random.randn(self.N, 1)
        self.r = self.mu + self.a_return
        # update standardized disturbances
        self.e = np.linalg.inv(self.D)@self.a_return
        
        return self.r
    
    def neg_log_likelihood(self, params):
        print("params: ", params)
        train = True
        log_likelihood = 0
        for t in range(self.T):
            # if t % 100 == 0:
            #     print(t)
            self.D_t(params=params, t=t, train=train)
            self.R_t(params=params, t=t, train=train)
            current_log_likelihood = self.N * np.log(2*np.pi) + 2*np.log(np.linalg.det(self.D)) + np.log(np.linalg.det(self.R)) + self.a_return.T@np.linalg.inv(self.D)@np.linalg.inv(self.R)@np.linalg.inv(self.D)@self.a_return
            log_likelihood += current_log_likelihood
        print("nll: ", log_likelihood/2)
        self.nll_losses.append(log_likelihood[0]/2)
        return log_likelihood/2

    
    def train(self):
        params = self.params
        results = minimize(self.neg_log_likelihood, params, method="trust-constr",
                           options={'disp': True}, constraints=self.constraints)
        return results.x

    def generate(self, S_0, num_points):
        prices = np.ones((self.N, num_points)) * S_0
        for t in range(num_points):
            self.D_t(params=self.params)
            self.R_t(params=self.params)
            r_t = self.r_t()
            prices[:, t+1:t+2] = prices[:, t:t+1]*np.exp(r_t)
        return prices

stocks = "^GDAXI ^GSPC"
data = yf.download(stocks, start="2000-09-10", end="2024-08-20", interval="1d")
log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
log_returns_np = log_returns.to_numpy().T

model = DCC_GARCH(log_returns_np)

params = model.train()
plt.plot(model.nll_losses)
plt.savefig("nll_losses.png")
plt.close()

x = model.generate(100, 252*5)
plt.plot(x.T)
plt.savefig("dcc_garch_test.png")
plt.close()