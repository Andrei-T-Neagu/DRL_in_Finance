import datetime as datetime
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Utils_general
import option_hedging.code_pytorch.DeepHedgingEnvironment as DeepHedgingEnvironment
import option_hedging.code_pytorch.old_models.DeepAgentTransformer as DeepAgentTransformer
import option_hedging.code_pytorch.old_models.DeepAgentLSTM as DeepAgentLSTM
import option_hedging.code_pytorch.old_models.DeepAgentGRU as DeepAgentGRU
import option_hedging.code_pytorch.old_models.DeepAgentLight as DeepAgentLight
import option_hedging.code_pytorch.old_models.DeepAgentTransformerLight as DeepAgentTransformerLight
import option_hedging.code_pytorch.old_models.DeepAgentLSTMLight as DeepAgentLSTMLight
import option_hedging.code_pytorch.old_models.DeepAgentGRULight as DeepAgentGRULight
from scipy.stats import wilcoxon
from scipy.stats import ttest_ind

nbs_point_traj = 9
time_period = "day"
batch_size = 256
train_size = 100000
test_size = 1000000
epochs = 100
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.0892, 0.1952]
S_0 = 1000
T = 1/252
alpha = 1.002
beta = 0.998
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 4
nbs_units = 256
num_heads = 8
lr = 0.0001
dropout = 0
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [-1, -1] 

generate_dataset = False
light = False
lr_schedule = True

if (option_type == 'call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

if generate_dataset:
    # Creating test dataset
    print("Generating Test Data Set")
    mu, sigma = params_vect
    N = nbs_point_traj - 1
    dt = T / N
    test_set = S_0 * torch.ones(int(test_size/batch_size), nbs_point_traj, batch_size)
    for i in range(int(test_size/batch_size)):
        S_t = S_0 * torch.ones(batch_size)
        for j in range(N):
            Z = torch.randn(batch_size)
            S_t = S_t * torch.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * Z)
            test_set[i, j+1, :] = S_t

    torch.save(test_set, "/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/test_set")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_set = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/test_set")
test_set = test_set.to(device)

path_prefix = "code_pytorch/errors_experiments/"

name_ffnn_light = 'code_pytorch/ffnn_model_light_' + str(nbs_point_traj)
name_lstm_light = 'code_pytorch/lstm_model_light_' + str(nbs_point_traj)
name_gru_light = 'code_pytorch/gru_model_light_' + str(nbs_point_traj)
name_transformer_light = 'code_pytorch/transformer_model_light_' + str(nbs_point_traj)

path_prefix = "code_pytorch/errors_experiments/"

name_ffnn = 'code_pytorch/ffnn_model_' + str(nbs_point_traj)
name_lstm = 'code_pytorch/lstm_model_' + str(nbs_point_traj)
name_gru = 'code_pytorch/gru_model_' + str(nbs_point_traj)
name_transformer = 'code_pytorch/transformer_model_' + str(nbs_point_traj)

"""MAX PARAMETERS TRANSFORMER:
batch_size = 512 | nbs_layers = 4 | nbs_units = 128
batch_size = 512 | nbs_layers = 2 | nbs_units = 256

batch_size = 256 | nbs_layers = 4 | nbs_units = 256
batch_size = 256 | nbs_layers = 2 | nbs_units = 256 

batch_size = 128 | nbs_layers = 10 | nbs_units = 256
batch_size = 128 | nbs_layers = 5 | nbs_units = 512
batch_size = 128 | nbs_layers = 2 | nbs_units = 1024

batch_size = 64 | nbs_layers = 10 | nbs_units = 512
batch_size = 64 | nbs_layers = 5 | nbs_units = 1024
"""


agent_trans_light = DeepAgentTransformerLight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, num_heads, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_transformer)

agent_trans = DeepAgentTransformer.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, num_heads, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_transformer)

agent_trans_light.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_transformer_light)
deltas_trans_light, hedging_err_trans_light, S_t_trans_light, V_t_trans_light, A_t_trans_light, B_t_trans_light = agent_trans_light.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_trans_light = np.square(np.where(hedging_err_trans_light > 0, hedging_err_trans_light, 0))
rsmse_trans_light = np.sqrt(np.mean(semi_square_hedging_err_trans_light))

agent_trans.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_transformer)
deltas_trans, hedging_err_trans, S_t_trans, V_t_trans, A_t_trans, B_t_trans = agent_trans.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_trans = np.square(np.where(hedging_err_trans > 0, hedging_err_trans, 0))
rsmse_trans = np.sqrt(np.mean(semi_square_hedging_err_trans))


agent_lstm_light = DeepAgentLSTMLight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_lstm)

agent_lstm = DeepAgentLSTM.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_lstm)

agent_lstm_light.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_lstm_light)
deltas_lstm_light, hedging_err_lstm_light, S_t_lstm_light, V_t_lstm_light, A_t_lstm_light, B_t_lstm_light = agent_lstm_light.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_lstm_light = np.square(np.where(hedging_err_lstm_light > 0, hedging_err_lstm_light, 0))
rsmse_lstm_light = np.sqrt(np.mean(semi_square_hedging_err_lstm_light))

agent_lstm.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_lstm)
deltas_lstm, hedging_err_lstm, S_t_lstm, V_t_lstm, A_t_lstm, B_t_lstm = agent_lstm.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_lstm = np.square(np.where(hedging_err_lstm > 0, hedging_err_lstm, 0))
rsmse_lstm = np.sqrt(np.mean(semi_square_hedging_err_lstm))


agent_gru_light = DeepAgentGRULight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_gru)

agent_gru = DeepAgentGRU.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_gru)

agent_gru_light.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_gru_light)
deltas_gru_light, hedging_err_gru_light, S_t_gru_light, V_t_gru_light, A_t_gru_light, B_t_gru_light = agent_gru_light.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_gru_light = np.square(np.where(hedging_err_gru_light > 0, hedging_err_gru_light, 0))
rsmse_gru_light = np.sqrt(np.mean(semi_square_hedging_err_gru_light))

agent_gru.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_gru)
deltas_gru, hedging_err_gru, S_t_gru, V_t_gru, A_t_gru, B_t_gru = agent_gru.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_gru = np.square(np.where(hedging_err_gru > 0, hedging_err_gru, 0))
rsmse_gru = np.sqrt(np.mean(semi_square_hedging_err_gru))


agent_light = DeepAgentLight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_ffnn)

agent = DeepHedgingEnvironment.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                nbs_shares, lambdas, name=name_ffnn)

agent_light.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_ffnn_light)
deltas_ffnn_light, hedging_err_ffnn_light, S_t_ffnn_light, V_t_ffnn_light, A_t_ffnn_light, B_t_ffnn_light = agent_light.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_ffnn_light = np.square(np.where(hedging_err_ffnn_light > 0, hedging_err_ffnn_light, 0))
rsmse_ffnn_light = np.sqrt(np.mean(semi_square_hedging_err_ffnn_light))

agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_ffnn)
deltas_ffnn, hedging_err_ffnn, S_t_ffnn, V_t_ffnn, A_t_ffnn, B_t_ffnn = agent.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_ffnn = np.square(np.where(hedging_err_ffnn > 0, hedging_err_ffnn, 0))
rsmse_ffnn = np.sqrt(np.mean(semi_square_hedging_err_ffnn))

print()

smse_trans = ttest_ind(semi_square_hedging_err_trans_light, semi_square_hedging_err_trans, equal_var=False, alternative="less").pvalue
smse_lstm = ttest_ind(semi_square_hedging_err_lstm_light, semi_square_hedging_err_lstm, equal_var=False, alternative="less").pvalue
smse_gru = ttest_ind(semi_square_hedging_err_gru_light, semi_square_hedging_err_gru, equal_var=False, alternative="less").pvalue
smse_ffnn = ttest_ind(semi_square_hedging_err_ffnn_light, semi_square_hedging_err_ffnn, equal_var=False, alternative="less").pvalue

print("|---------------------------RSMSE of Light Model------------------------|")
print("|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|")
print("|-----------------------|---------------|---------------|---------------|")
print("|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|".format(rsmse_trans_light, rsmse_gru_light, rsmse_lstm_light, rsmse_ffnn_light))
print("|-----------------------|---------------|---------------|---------------|")
print("|---------------------------RSMSE of Normal Model-----------------------|")
print("|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|")
print("|-----------------------|---------------|---------------|---------------|")
print("|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|".format(rsmse_trans, rsmse_gru, rsmse_lstm, rsmse_ffnn))
print("|-----------------------|---------------|---------------|---------------|")
print("|---------------------T-test for smaller SMSE (Light vs Regular model)--------------------------|")
print("|\t\t\t|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|")
print("|-----------------------|-----------------------|---------------|---------------|---------------|")
print("|\tp-values:\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|".format(smse_trans, smse_gru, smse_lstm, smse_ffnn))
print("|-----------------------|-----------------------|---------------|---------------|---------------|")

with open(path_prefix + "ModelComparison_" + str(nbs_point_traj) + ".txt", "w") as comparison_file:
    # Writing data to a file
    comparison_file.write("|---------------------------RSMSE of Light Model------------------------|\n")
    comparison_file.write("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\n")
    comparison_file.write("|-----------------------|---------------|---------------|---------------|\n")
    comparison_file.write("|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\n".format(rsmse_trans_light, rsmse_gru_light, rsmse_lstm_light, rsmse_ffnn_light))
    comparison_file.write("|-----------------------|---------------|---------------|---------------|\n")
    comparison_file.write("|---------------------------RSMSE of Normal Model-----------------------|\n")
    comparison_file.write("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\n")
    comparison_file.write("|-----------------------|---------------|---------------|---------------|\n")
    comparison_file.write("|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\n".format(rsmse_trans, rsmse_gru, rsmse_lstm, rsmse_ffnn))
    comparison_file.write("|-----------------------|---------------|---------------|---------------|\n")
    comparison_file.write("|----------------------T-test for smaller SMSE (Light vs Regular model)-------------------------|\n")
    comparison_file.write("|\t\t\t\t\t\t|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\n")
    comparison_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|\n")
    comparison_file.write("|\t\tp-values:\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\n".format(smse_trans, smse_gru, smse_lstm, smse_ffnn))
    comparison_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|\n")

print()

fig = plt.figure(figsize=(15, 10))
plt.hist(semi_square_hedging_err_gru_light, bins=50, label="GRU")
plt.xlabel('Semi Squared Errors')
plt.ylabel('Frequency')
plt.legend()
plt.title("Semi Squared Errors for GRU Light - " + str(nbs_point_traj))
plt.savefig(path_prefix + "hedging_errors/Semi_Square_Error_GRU_Light" + str(nbs_point_traj) + ".png")

max_error_index = np.argmax(hedging_err_ffnn)
ratios_DH_worst, hedging_err_DH_worst = Utils_general.delta_hedge_res(S_t_ffnn[:,max_error_index:max_error_index+1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
ratios_ffnn_worst = deltas_ffnn[:,max_error_index,0]
ratios_lstm_worst = deltas_lstm[:,max_error_index,0]
ratios_gru_worst = deltas_gru[:,max_error_index,0]
ratios_trans_worst = deltas_trans[:, max_error_index,0]

fig = plt.figure(figsize=(15, 10))
plt.plot(S_t_ffnn[:,max_error_index], label="Stock prices path")
plt.xlabel('Time Steps')
plt.ylabel('Stock Prices')
plt.grid()
plt.legend()
plt.title("Paths of stock prices for worst error scenario - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_stock_prices_worst_losses_{time_period}.jpg".format(time_period=time_period))

fig = plt.figure(figsize=(15, 10))
plt.plot(ratios_DH_worst, label="delta hedge")
plt.plot(ratios_ffnn_worst, label="FFNN")
plt.plot(ratios_lstm_worst, label="LSTM")
plt.plot(ratios_gru_worst, label="GRU")
plt.plot(ratios_trans_worst, label="Transformer")
plt.xlabel('Time Steps')
plt.ylabel('Hedge Ratios')
plt.grid()
plt.legend()
plt.title("Paths of hedge ratios of models for worst error scenario - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_ratios_models_worst_error_{time_period}.jpg".format(time_period=time_period))

min_error_index = np.argmin(hedging_err_ffnn)
ratios_DH_best, hedging_err_DH_best = Utils_general.delta_hedge_res(S_t_ffnn[:,min_error_index:min_error_index+1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
ratios_ffnn_best = deltas_ffnn[:,min_error_index,0]
ratios_lstm_best = deltas_lstm[:,min_error_index,0]
ratios_gru_best = deltas_gru[:,min_error_index,0]
ratios_trans_best = deltas_trans[:, min_error_index,0]

fig = plt.figure(figsize=(15, 10))
plt.plot(S_t_ffnn_light[:,min_error_index], label="Stock prices path")
plt.xlabel('Time Steps')
plt.ylabel('Stock Prices')
plt.grid()
plt.legend()
plt.title("Paths of stock prices for best error scenario - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_stock_prices_best_error_{time_period}.jpg".format(time_period=time_period))

fig = plt.figure(figsize=(15, 10))
plt.plot(ratios_DH_best, label="delta hedge")
plt.plot(ratios_ffnn_best, label="FFNN")
plt.plot(ratios_lstm_best, label="LSTM")
plt.plot(ratios_gru_best, label="GRU")
plt.plot(ratios_trans_best, label="Transformer")
plt.xlabel('Time Steps')
plt.ylabel('Hedge Ratios')
plt.grid()
plt.legend()
plt.title("Paths of hedge ratios of models for best error scenario - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_ratios_models_best_error_{time_period}.jpg".format(time_period=time_period))

#LIGHT MODELS

max_error_index_light = np.argmax(hedging_err_ffnn_light)
ratios_DH_worst_light, hedging_err_DH_worst_light = Utils_general.delta_hedge_res(S_t_ffnn[:,max_error_index_light:max_error_index_light+1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
ratios_ffnn_worst_light = deltas_ffnn_light[:,max_error_index_light,0]
ratios_lstm_worst_light = deltas_lstm_light[:,max_error_index_light,0]
ratios_gru_worst_light = deltas_gru_light[:,max_error_index_light,0]
ratios_trans_worst_light = deltas_trans_light[:, max_error_index_light,0]

fig = plt.figure(figsize=(15, 10))
plt.plot(S_t_ffnn_light[:,max_error_index_light], label="Stock prices path")
plt.xlabel('Time Steps')
plt.ylabel('Stock Prices')
plt.grid()
plt.legend()
plt.title("Paths of stock prices for worst error scenario - Light models - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_stock_prices_worst_losses_light_{time_period}.jpg".format(time_period=time_period))

fig = plt.figure(figsize=(15, 10))
plt.plot(ratios_DH_worst_light, label="delta hedge")
plt.plot(ratios_ffnn_worst_light, label="FFNN")
plt.plot(ratios_lstm_worst_light, label="LSTM")
plt.plot(ratios_gru_worst_light, label="GRU")
plt.plot(ratios_trans_worst_light, label="Transformer")
plt.xlabel('Time Steps')
plt.ylabel('Hedge Ratios')
plt.grid()
plt.legend()
plt.title("Paths of hedge ratios of models for worst error scenario - Light models - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_ratios_models_worst_error_light_{time_period}.jpg".format(time_period=time_period))

min_error_index_light = np.argmin(hedging_err_ffnn_light)
ratios_DH_best_light, hedging_err_DH_best_light = Utils_general.delta_hedge_res(S_t_ffnn_light[:,min_error_index_light:min_error_index_light+1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
ratios_ffnn_best_light = deltas_ffnn_light[:,min_error_index_light,0]
ratios_lstm_best_light = deltas_lstm_light[:,min_error_index_light,0]
ratios_gru_best_light = deltas_gru_light[:,min_error_index_light,0]
ratios_trans_best_light = deltas_trans_light[:, min_error_index_light,0]

fig = plt.figure(figsize=(15, 10))
plt.plot(S_t_ffnn_light[:,min_error_index_light], label="Stock prices path")
plt.xlabel('Time Steps')
plt.ylabel('Stock Prices')
plt.grid()
plt.legend()
plt.title("Paths of stock prices for best error scenario - Light models - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_stock_prices_best_error_light_{time_period}.jpg".format(time_period=time_period))

fig = plt.figure(figsize=(15, 10))
plt.plot(ratios_DH_best_light, label="delta hedge")
plt.plot(ratios_ffnn_best_light, label="FFNN")
plt.plot(ratios_lstm_best_light, label="LSTM")
plt.plot(ratios_gru_best_light, label="GRU")
plt.plot(ratios_trans_best_light, label="Transformer")
plt.xlabel('Time Steps')
plt.ylabel('Hedge Ratios')
plt.grid()
plt.legend()
plt.title("Paths of hedge ratios of models for best error scenario - Light models - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
plt.savefig(path_prefix + "path_of_ratios_models_best_error_light_{time_period}.jpg".format(time_period=time_period))