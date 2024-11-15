import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pickle
import yfinance as yf
from arch import arch_model
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from GARCH import GARCH

stock = "^GSPC"
garch_type = "vanilla"
start="1986-12-31" 
end="2010-04-01"
interval="1d"

model = GARCH(stock=stock, start=start, end=end, interval=interval, type=garch_type)
# model = GARCH(stock=stock, start="2024-01-01", end="2024-09-20", interval="1h", type="gjr")

"""Training"""
params = model.train(save_params=True)
plt.figure(figsize=(12,6))
nll_losses = np.array(model.nll_losses)
plt.plot(nll_losses[nll_losses <= nll_losses[0]])
plt.xlabel("Function Evaluations")
plt.ylabel("Negative Log Likelihood")
plt.savefig("garch_nll_losses.png")
plt.close()

"""Generating"""
x = model.generate(S_0=100, batch_size=2**17, num_points=252*36, load_params=True)
print(x.shape)
log_returns = np.log(x/np.roll(x,1,axis=1))[:,1:]
mu = np.mean(log_returns,axis=1)                                                                  # expected value of r_t
print("DATA MU: ", np.mean(model.r_data/100))
print("GENERATED MU: ", np.mean(mu))
h = np.var(log_returns,axis=1)                                                                    # conditional variance of a_t
print("DATA VAR:", np.var(model.r_data/100))
print("GENERATED VAR:", np.mean(h))
plt.figure(figsize=(12,6))
plt.plot(x[0].T)
plt.xlabel("Timesteps")
plt.ylabel("Prices")
plt.legend([stock])
plt.savefig("garch_test.png")
plt.close()

"""Comparison of parameters with arch library"""
market_data = yf.download(stock, start=start, end=end, interval=interval,timeout=60)
log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()

mu = log_returns.mean() * 252
sigma = log_returns.std() * np.sqrt(252)
print(f"Estimated Annualized Mu (drift) from data: {mu}")
print(f"Estimated Annulaized Sigma (volatility) from data: {sigma}")

data = log_returns.to_numpy()*100
if garch_type == "vanilla":
    model1 = arch_model(data.T, vol='Garch', p=1, q=1)
elif garch_type == "gjr":
    model1 = arch_model(data.T, vol='Garch', p=1, q=1, o=1)
elif garch_type == "e":
    model1 = arch_model(data.T, vol='EGarch', p=1, q=1, o=1)
result1 = model1.fit()
print(result1.summary())
model.print_params()

"""Comparison of parameters with rugarch R library"""
returns_matrix = numpy2ri.py2rpy(data)

# Import the necessary R packages
rugarch = importr('rugarch')

if garch_type == "gjr":
    model_type = "gjrGARCH"
elif garch_type == "e":
    model_type = "eGARCH"
elif garch_type == "vanilla":
    model_type = "sGARCH"               # Vanilla GARCH

# Define the R code for specifying and fitting the GJR-GARCH model
r_code = f"""
library(rugarch)

# Step 1: Define the GJR-GARCH model specification
gjr_spec <- ugarchspec(variance.model = list(model = "{model_type}", garchOrder = c(1, 1)),
                    mean.model = list(armaOrder = c(0, 0)),
                    distribution.model = "norm")

# Step 2: Fit the GJR-GARCH model to the returns data
gjr_fit <- ugarchfit(spec = gjr_spec, data = returns_data)

# Output the fit result
gjr_fit
"""

# Execute the R code, passing the 'returns_r' object as 'returns_data'
robjects.globalenv['returns_data'] = returns_matrix
garch_fit = robjects.r(r_code)

# Print or inspect the model fit result
print(garch_fit)
model.print_params()