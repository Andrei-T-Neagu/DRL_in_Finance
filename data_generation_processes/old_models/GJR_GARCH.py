import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def GJR_GARCH(S_0, mu, gamma, omega, alpha, beta, N):
    prices = np.ones(N) * S_0
    sigma2 = omega / (1-alpha*(1+gamma**2)-beta)
    print(sigma2)
    for t in range(N-1):
        epsilon = np.random.randn(1)
        r_t = mu + np.sqrt(sigma2) * epsilon
        sigma2 = omega + alpha * sigma2 * (abs(epsilon) - gamma * epsilon)**2 + beta * sigma2
        prices[t+1] = prices[t]*np.exp(r_t)[0]
    return prices

prices = GJR_GARCH(100, 0.0002871, 0.6028, 0.000001795, 0.0540, 0.9105, 252*5)
plt.figure(figsize=(12,6))
plt.plot(prices)
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.savefig("gjr_garch_test.png")
plt.close()

stocks = "^GSPC"
data = yf.download(stocks, start="1986-12-31", end="2010-04-01", interval="1d")
log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
log_returns_np = log_returns.to_numpy().T

a_data = log_returns_np - np.mean(log_returns_np)
print(np.var(a_data))
