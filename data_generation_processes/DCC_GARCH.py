import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
        self.r_data = data                                                      #(N, T)
        self.mu = np.mean(self.r_data, axis=1, keepdims=True)                        #(N, 1) 
        self.a_data = self.r_data - np.mean(self.r_data, axis=1, keepdims=True)           #(N, T)
        self.H = np.cov(self.a_data)                                       #(N, N)
        self.H_factor = np.linalg.cholesky(self.H)                        #(N, N)
        self.D = np.diag(np.std(self.r_data, axis=1))                           #(N, N)
        self.R = np.corrcoef(self.a_data)                                  #(N, N)
        self.alpha0 = np.random.randn(self.N, 1)*0.1                            #(N, 1)
        self.alpha = np.random.randn(self.N, 1)*0.1                             #(N, 1)
        self.beta = np.random.randn(self.N, 1)*0.1                              #(N, 1)
        self.a_return = self.H_factor@np.random.randn(self.N, 1)      #(N, 1)
        self.r = self.mu + self.a_return     
        self.e = np.linalg.inv(self.D)@self.a_return                                #(N, 1)
        self.e_data = np.linalg.inv(self.D)@self.a_data                             #(N, T)
        self.Q_bar = self.e_data@self.e_data.T/self.T                               #(N, N)
        self.a = np.random.randn(1)*0.1
        self.b = np.random.randn(1)*0.1                                             
        self.Q = self.Q_bar                                                         #(N, N)

    def D_t(self):
        """
            h_{n,t} = alpha0_{n} + alpha_{n} a_{n, t-1}^2 + beta_{n} h_{n, t-1}
        """
        h_vector = np.expand_dims(np.diag(self.D_prev)**2,axis=1)
        new_h_vector = self.alpha0 + self.alpha * self.a_return**2 + self.beta * h_vector
        return np.diag(np.sqrt(new_h_vector.flatten()))
    
    def R_t(self):
        """
            e_t = D_t^{-1} a_t \sim N(0, R_t)
            R_t = Q_t^{*-1} Q_t Q_t^{*-1}
            Q_t = (1 - a - b) Q_bar + a e_{t-1} e_{t-1}^T + b Q_{t-1}

        """
        self.Q = (1 - self.a - self.b) * self.Q_bar + self.a * self.e@self.e.T + self.b * self.Q
        Q_diag_inv = np.linalg.inv(np.diag(np.diag(np.sqrt(self.Q))))
        self.R = Q_diag_inv@self.Q@Q_diag_inv
        return

    def log_likelihood(self):
        return 

e = np.array([[3,7,2],[8,1,5],[2,5,4]])
print(e)
print()
Q_bar = e@e.T/3
Q_bar_it = np.zeros((3,3))
print(e[:, 0:1])
print()
for i in range(Q_bar_it.shape[1]):
    Q_bar_it = Q_bar_it + e[:, i:i+1]@e[:, i:i+1].T
print(Q_bar_it/3)
print(Q_bar)