import numpy as np
import matplotlib.pyplot as plt

def GJR_GARCH(S_0, mu, gamma, omega, alpha, beta, N):
    prices = np.ones(N) * S_0
    sigma2 = omega / (1-alpha*(1+gamma**2)-beta)
    for t in range(N-1):
        epsilon = np.random.randn(1)
        r_t = mu + np.sqrt(sigma2) * epsilon
        sigma2 = omega + alpha * sigma2 * (abs(epsilon) - gamma * epsilon)**2 + beta * sigma2
        prices[t+1] = prices[t]*np.exp(r_t)[0]
    return prices

prices = GJR_GARCH(100, 0.0002871, 0.6028, 0.000001795, 0.0540, 0.9105, 252)
plt.plot(prices)
plt.savefig("gjr_garch_test.png")
