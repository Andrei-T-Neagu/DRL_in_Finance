import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

T = 1/252
N = 8
alpha = 1.001
beta = 0.999
delta_t = T / N
lambda_a = -math.log(0.5)/delta_t
lambda_b = -math.log(0.5)/delta_t
# lambda_a = 0
# lambda_b = 0
A_t = 0
B_t = 0
strike = 1000

A_t = (A_t + 0.5) * math.exp(-lambda_a * delta_t)
A_t = 0
print("A_t: ", A_t)
price_buying = strike * ((1 + 1 + A_t) ** alpha - (1 + A_t) ** alpha)

B_t = (B_t + 0.5) * math.exp(-lambda_b * delta_t)
B_t = 0
print("B_t: ", B_t)
price_selling = strike * ((1 + 1 + B_t) ** beta - (1 + B_t) ** beta)

print("price_buying: ", price_buying)
print("price_selling: ", price_selling)
print("strike_price: ", strike)
print("k for buying: ", (price_buying-strike)/strike)
print("k for selling: ", -(price_selling-strike)/strike)

# np.random.seed(0)
# var1 = np.random.randn(10000)*1.1
# var1_smse = np.square(np.where(var1 > 0, var1, 0))

# np.random.seed(1)
# var2 = np.random.randn(20000)
# var2_smse = np.square(np.where(var2 > 0, var2, 0))

# t_test = ttest_ind(var2, var1, equal_var=False, alternative="less")
# t_test_smse = ttest_ind(var2_smse, var1_smse, equal_var=False, alternative="less")

# var1_smse_mean = np.mean(var1_smse)
# var2_smse_mean = np.mean(var2_smse)

# var1_smse_stde = np.std(var1_smse, ddof=1)/np.sqrt(10000)
# var2_smse_stde = np.std(var2_smse, ddof=1)/np.sqrt(20000)

# t = (var2_smse_mean-var1_smse_mean)/(np.sqrt(np.square(var2_smse_stde)+np.square(var1_smse_stde)))

# var1_rsmse = np.sqrt(var1_smse_mean)
# var2_rsmse = np.sqrt(var2_smse_mean)

# var1_rsmse_stde = np.std(var1_smse, ddof=1)/np.sqrt(10000)
# var2_rsmse_stde = np.std(var2_smse, ddof=1)/np.sqrt(20000)

# t_r = (var2_rsmse-var1_rsmse)/(np.sqrt(np.square(var2_smse_stde)+np.square(var1_smse_stde)))

# print(t_test)
# print(t)
# print(t_r)
# print(t_test_smse)

# print("|-----------------------------------------------------------------------------------------------Comparison of Mean Hedging Error-----------------------------------------------------------------------------------------------|")
# print("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t\t|\t\tFFNN\t\t|\t\tDelta Hedge\t\t|\t\tLeland Delta Hedge\t\t|")
# print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|----------------------------------------------|")
# print("|\t{:.4f} +- {:.4f}\t\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t\t|".format(np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()))
# print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|----------------------------------------------|")


# lstm_losses = np.exp(-np.linspace(-5, -4, 100))*np.random.rand(100)
# ffnn_losses = np.exp(-np.linspace(-5, -4, 100))*np.random.rand(100)
# trans_losses = np.exp(-np.linspace(-5, -4, 100))*np.random.rand(100)
# gru_losses = np.exp(-np.linspace(-5, -4, 100))*np.random.rand(100)
# print(lstm_losses)
# log_epoch_losses_fig = plt.figure(figsize=(15, 10))
# plt.plot(lstm_losses, label="LSTM")
# plt.plot(ffnn_losses, label="FFNN")
# plt.plot(trans_losses, label="Transformer")
# plt.plot(gru_losses, label="GRU")
# plt.yscale("log")
# plt.legend()
# plt.savefig("code_pytorch/log_losses_test.png")

# value = np.ones(1)
# print(str(value.item()))