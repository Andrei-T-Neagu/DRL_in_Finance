import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import yfinance as yf
import pickle
import GARCH
from arch import arch_model
import mgarch
from mgarch import mgarch
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from DCC_GARCH_BATCH import DCC_GARCH

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

# stocks = "^GDAXI ^IXIC ^GSPC ^DJI ^RUT ^N225 ^HSI ^NYA ^FCHI"
stocks = "^IXIC ^GSPC ^DJI"

garch_type = "gjr"
start="2000-10-24"
end="2024-09-24"
interval="1d"

dcc_garch_model = DCC_GARCH(stocks=stocks, start=start, end=end, interval=interval, type=garch_type)

sorted_stocks_list = sorted(list(stocks.split(" ")))
"""Training"""
params = dcc_garch_model.train_R(save_params=True)
plt.figure(figsize=(12,6))
nll_losses = np.array(dcc_garch_model.nll_losses)
plt.plot(nll_losses[nll_losses <= nll_losses[0]])
plt.xlabel("Function Evaluations")
plt.ylabel("Negative Log Likelihood")
plt.savefig("dcc_nll_losses.png")
plt.close()

"""Generating"""
num_points=252*5

plt.figure(figsize=(12,6))
x = dcc_garch_model.generate(S_0=100, batch_size=2**1, num_points=252*100, load_params=True)
log_returns = np.log(x/np.roll(x,1,axis=2))[0,:,1:]*100
a = log_returns - np.mean(log_returns, axis=1, keepdims=True)
R = np.corrcoef(a)
print("Correlation matrix from generated data:\n", R)
print("Correlation matrix from real data:\n", np.corrcoef(dcc_garch_model.a_data))
plt.plot(x[0].T)
plt.xlabel("Timesteps")
plt.ylabel("Prices")
plt.legend(sorted_stocks_list)
plt.savefig("dcc_garch_test.png")
plt.close()

"""Comparison of parameters with mgarch library"""
market_data = yf.download(stocks, start=start, end=end, interval=interval, timeout=60)
log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
data = log_returns.to_numpy()*100                                        #(T, N) Dataset of log returns

model = mgarch()
result = model.fit(data)

print(result)

"""Comparison of parameters with rmgarch R library"""
numpy2ri.activate()

# Enable the automatic conversion between pandas and R dataframes
returns_matrix = numpy2ri.py2rpy(data)

# Import the necessary R packages
rugarch = importr('rugarch')
rmgarch = importr('rmgarch')

if garch_type == "gjr":
    model_type = "gjrGARCH"
elif garch_type == "e":
    model_type = "eGARCH"
elif garch_type == "vanilla":
    model_type = "sGARCH"               # Vanilla GARCH

garch_spec_list = []
for i in range(len(sorted_stocks_list)):
    garch_spec_list.append(f'garch_spec{i} <- ugarchspec(variance.model = list(model = "{model_type}", garchOrder = c(1, 1)),'
                           ' mean.model = list(armaOrder = c(0, 0)))')

garch_spec_str = ', '.join([f'garch_spec{i}' for i in range(len(sorted_stocks_list))])

# Define the R code for specifying and fitting the GJR-GARCH model
r_code = f"""
library(rugarch)
library(rmgarch)

# Step 1: Define univariate GJR-GARCH models for each asset using ugarchspec
{'\n'.join(garch_spec_list)}
                        
# Step 2: Combine them into a multispec object
uspec <- multispec(list({garch_spec_str}))

# Step 3: Define the DCC-GARCH model using dccspec
dcc_spec <- dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = "mvnorm")

# Output the DCC specification object to Python
dcc_spec
"""

# Execute the R code from Python
dcc_spec = robjects.r(r_code)

# Now `dcc_spec` contains the DCC-GARCH model specification
print(dcc_spec)

# Fit the DCC-GARCH model
dcc_fit = rmgarch.dccfit(dcc_spec, data=returns_matrix)

# Print the summary of the DCC-GARCH fit
robjects.r('print')(dcc_fit)
dcc_garch_model.print_params()