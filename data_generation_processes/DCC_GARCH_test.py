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
stocks = "^IXIC ^GSPC"
garch_type = "vanilla"
start="2000-10-24"
end="2024-09-24"
interval="1d"

dcc_garch_model = DCC_GARCH(stocks=stocks, start=start, end=end, interval=interval, type=garch_type)

"""Training"""
params = dcc_garch_model.train_R(save_params=True)
plt.figure(figsize=(12,6))
plt.plot(dcc_garch_model.nll_losses)
plt.xlabel("Function Evaluations")
plt.ylabel("Negative Log Likelihood")
plt.savefig("dcc_nll_losses.png")
plt.close()

"""Generating"""
num_points=252*5

plt.figure(figsize=(12,6))
x = dcc_garch_model.generate(S_0=100, batch_size=2**2, num_points=252*5, load_params=True)
plt.plot(x[0].T)
plt.xlabel("Timesteps")
plt.ylabel("Prices")
plt.legend(list(stocks.split(" ")))
plt.savefig("dcc_garch_test.png")
plt.close()

"""Comparison of parameters with mgarch library"""
market_data = yf.download(stocks, start=start, end=end, interval=interval)
log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
data = log_returns.to_numpy()                                        #(T, N) Dataset of log returns

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
    r_code = """
    library(rugarch)
    library(rmgarch)

    # Step 1: Define univariate GJR-GARCH models for each asset using ugarchspec
    gjr_spec1 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
                            mean.model = list(armaOrder = c(0, 0)))

    gjr_spec2 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
                            mean.model = list(armaOrder = c(0, 0)))

    # gjr_spec3 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # gjr_spec4 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # gjr_spec5 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # gjr_spec6 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # gjr_spec7 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))

    # gjr_spec8 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))

    # gjr_spec9 <- ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # Step 2: Combine them into a multispec object
    uspec <- multispec(list(gjr_spec1, gjr_spec2))
    # uspec <- multispec(list(gjr_spec1, gjr_spec2, gjr_spec3, gjr_spec4, gjr_spec5, gjr_spec6, gjr_spec7, gjr_spec8, gjr_spec9))

    # Step 3: Define the DCC-GARCH model using dccspec
    dcc_spec <- dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = "mvnorm")

    # Output the DCC specification object to Python
    dcc_spec
    """
else:
    r_code = """
    library(rugarch)
    library(rmgarch)

    # Step 1: Define univariate GJR-GARCH models for each asset using ugarchspec
    garch_spec1 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                            mean.model = list(armaOrder = c(0, 0)))

    garch_spec2 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                            mean.model = list(armaOrder = c(0, 0)))

    # garch_spec3 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # garch_spec4 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # garch_spec5 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # garch_spec6 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # garch_spec7 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))

    # garch_spec8 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))

    # garch_spec9 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    #                         mean.model = list(armaOrder = c(0, 0)))
                            
    # Step 2: Combine them into a multispec object
    uspec <- multispec(list(garch_spec1, garch_spec2))
    # uspec <- multispec(list(garch_spec1, garch_spec2, garch_spec3, garch_spec4, garch_spec5, garch_spec6, garch_spec7, garch_spec8, garch_spec9))

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