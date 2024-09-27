import datetime as datetime
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Utils_general
import DeepAgent
import DeepAgentTransformer
import DeepAgentLSTM
import DeepAgentGRU
import DeepAgentLight
import DeepAgentTransformerLight
import DeepAgentLSTMLight
import DeepAgentGRULight
import sys
sys.path.insert(0,".")
from data_generation_processes.GARCH import GARCH
from scipy.stats import ttest_ind
from scipy.stats import f
global_path_prefix = "/home/a_eagu/DRL_in_Finance/option_hedging/code_pytorch/"

nbs_point_traj = 253
time_period = "day"
T = 252/252
alpha = 1.010
beta = 0.990

batch_size = 512
train_size = 2**15
test_size = 2**15
epochs = 1
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.0892, 0.1952]
S_0 = 100
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 100
nbs_layers = 4
nbs_units = 256
num_heads = 8
lr = 0.0001
dropout = 0
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [-1, -1] 
stock = "^GSPC"

light = False
lr_schedule = False

# Initializing the inital option price using black-scholes option pricing
if (option_type == 'call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

# Initialize the garch model
garch_model = GARCH(stock=stock, start="1986-12-31", end="2010-04-01", interval= "1d", type="gjr")

# Creating Black-Scholes datasets
def generate_BS_dataset(dataset_type="train_set", size=train_size):
    print("Generating Black-Scholes Data Set")
    mu, sigma = params_vect
    N = nbs_point_traj - 1
    dt = T / N
    dataset = S_0 * torch.ones(size, nbs_point_traj)
    for i in range(N):
        print("timestep: ", i)
        Z = torch.randn(size)
        dataset[:, i+1] = dataset[:, i] * torch.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * Z)

    torch.save(dataset, global_path_prefix + str(dataset_type))

# Training the GARCH model
def train_garch():
    params = garch_model.train(save_params=True)

# Generating the GARCH datasets
def generate_garch_dataset(dataset_type="train_set", size=train_size):
    dataset = garch_model.generate(S_0=S_0, batch_size=size, num_points=nbs_point_traj, load_params=True)
    dataset = torch.from_numpy(dataset).to(torch.float)
    
    torch.save(dataset, global_path_prefix + str(dataset_type))

"""Training the garch model and generating the datasets"""
train_garch()
generate_garch_dataset(dataset_type="train_set", size=train_size)
generate_garch_dataset(dataset_type="test_set", size=test_size)

# Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the training and testing datasets
train_set = torch.load(global_path_prefix + "train_set")
test_set = torch.load(global_path_prefix + "test_set")

# filepaths initialization
if light:
    errors_path_prefix = global_path_prefix + "errors_experiments/light/"

    name_ffnn = global_path_prefix + 'ffnn_model_light_' + str(nbs_point_traj)
    name_lstm = global_path_prefix + 'lstm_model_light_' + str(nbs_point_traj)
    name_gru = global_path_prefix + 'gru_model_light_' + str(nbs_point_traj)
    name_transformer = global_path_prefix + 'transformer_model_light_' + str(nbs_point_traj)
else:
    errors_path_prefix = global_path_prefix + "errors_experiments/"

    name_ffnn = global_path_prefix + 'ffnn_model_' + str(nbs_point_traj)
    name_lstm = global_path_prefix + 'lstm_model_' + str(nbs_point_traj)
    name_gru = global_path_prefix + 'gru_model_' + str(nbs_point_traj)
    name_transformer = global_path_prefix + 'transformer_model_' + str(nbs_point_traj)

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

# # Initialize Transformer model
# if light:
#     agent_trans = DeepAgentTransformerLight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
#                  loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, num_heads, lr, dropout, prepro_stock,
#                  nbs_shares, lambdas, name=name_transformer)
# else:
#     agent_trans = DeepAgentTransformer.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
#                  loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, num_heads, lr, dropout, prepro_stock,
#                  nbs_shares, lambdas, name=name_transformer)

# # Train and test Transformer model
# print("START TRANSFORMER")
# all_losses_trans, trans_losses = agent_trans.train(train_set=train_set, train_size = train_size, epochs=epochs, lr_schedule=lr_schedule)
# print("DONE TRANSFORMER")
# agent_trans.model = torch.load(name_transformer)
# deltas_trans, hedging_err_trans, S_t_trans, V_t_trans, A_t_trans, B_t_trans = agent_trans.test(test_size=test_size, test_set=test_set)
# semi_square_hedging_err_trans = np.square(np.where(hedging_err_trans > 0, hedging_err_trans, 0))
# smse_trans = np.mean(semi_square_hedging_err_trans)
# rsmse_trans = np.sqrt(np.mean(semi_square_hedging_err_trans))

# Initialize LSTM model
if light:
    agent_lstm = DeepAgentLSTMLight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                    loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                    nbs_shares, lambdas, name=name_lstm)
else:
    agent_lstm = DeepAgentLSTM.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                    loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                    nbs_shares, lambdas, name=name_lstm)

# Train and test LSTM model
print("START LSTM")
all_losses_lstm, lstm_losses = agent_lstm.train(train_set=train_set, train_size = train_size, epochs=epochs, lr_schedule = lr_schedule)
print("DONE LSTM")
agent_lstm.model = torch.load(name_lstm)
deltas_lstm, hedging_err_lstm, S_t_lstm, V_t_lstm, A_t_lstm, B_t_lstm = agent_lstm.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_lstm = np.square(np.where(hedging_err_lstm > 0, hedging_err_lstm, 0))
smse_lstm = np.mean(semi_square_hedging_err_lstm)
rsmse_lstm = np.sqrt(np.mean(semi_square_hedging_err_lstm))

# Initialize GRU model
if light:
    agent_gru = DeepAgentGRULight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                    loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                    nbs_shares, lambdas, name=name_gru)
else:
    agent_gru = DeepAgentGRU.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                    loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                    nbs_shares, lambdas, name=name_gru)

# Train and test GRU model
print("START GRU")
all_losses_gru, gru_losses = agent_gru.train(train_set=train_set, train_size = train_size, epochs=epochs, lr_schedule=lr_schedule)
print("DONE GRU")
agent_gru.model = torch.load(name_gru)
deltas_gru, hedging_err_gru, S_t_gru, V_t_gru, A_t_gru, B_t_gru = agent_gru.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_gru = np.square(np.where(hedging_err_gru > 0, hedging_err_gru, 0))
smse_gru = np.mean(semi_square_hedging_err_gru)
rsmse_gru = np.sqrt(np.mean(semi_square_hedging_err_gru))

# Initialize FFNN model
if light:
    agent = DeepAgentLight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                    loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                    nbs_shares, lambdas, name=name_ffnn)
else:
    agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                 nbs_shares, lambdas, name=name_ffnn)

# Train and test FFNN model
print("START FFNN")
all_losses_ffnn, ffnn_losses = agent.train(train_set=train_set, train_size = train_size, epochs=epochs, lr_schedule=lr_schedule)
print("DONE FFNN")
agent.model = torch.load(name_ffnn)
deltas_ffnn, hedging_err_ffnn, S_t_ffnn, V_t_ffnn, A_t_ffnn, B_t_ffnn = agent.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_ffnn = np.square(np.where(hedging_err_ffnn > 0, hedging_err_ffnn, 0))
smse_ffnn = np.mean(semi_square_hedging_err_ffnn)
rsmse_ffnn = np.sqrt(np.mean(semi_square_hedging_err_ffnn))

"""Print neural netwok models performance statistics"""

# print(" ----------------- ")
# print(" Deep Hedging %s TRANSFORMER Results" % (loss_type))
# print(" ----------------- ")
# Utils_general.print_stats(hedging_err_trans, deltas_trans, loss_type, "Deep hedge - TRANSFORMER - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Deep Hedging %s GRU Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_gru, deltas_gru, loss_type, "Deep hedge - GRU - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Deep Hedging %s LSTM Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_lstm, deltas_lstm, loss_type, "Deep hedge - LSTM - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Deep Hedging %s FFNN Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_ffnn, deltas_ffnn, loss_type, "Deep hedge - FFNN - %s" % (loss_type), V_0)

test_set = test_set.detach().cpu().numpy().T

"""Print baseline models performance statistics"""

print(" ----------------- ")
print(" Delta Hedging Results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(test_set, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)
semi_square_hedging_err_DH = np.square(np.where(hedging_err_DH > 0, hedging_err_DH, 0))
smse_DH = np.mean(semi_square_hedging_err_DH)
rsmse_DH = np.sqrt(np.mean(semi_square_hedging_err_DH))


print(" ----------------- ")
print("Leland Delta Hedging Results")
print(" ----------------- ")
deltas_DH_leland, hedging_err_DH_leland = Utils_general.delta_hedge_res(test_set, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas, Leland=True)
Utils_general.print_stats(hedging_err_DH_leland, deltas_DH_leland, "Leland delta hedge", "Leland delta hedge", V_0)
semi_square_hedging_err_DH_leland = np.square(np.where(hedging_err_DH_leland > 0, hedging_err_DH_leland, 0))
smse_DH_leland = np.mean(semi_square_hedging_err_DH_leland)
rsmse_DH_leland = np.sqrt(np.mean(semi_square_hedging_err_DH_leland))


print()

# print("|------------------------------------------------------Comparison of RSMSE-----------------------------------------------------|")
# print("|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|\tDelta Hedge\t|\tLeland Delta Hedge\t|")
# print("|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(rsmse_trans, rsmse_gru, rsmse_lstm, rsmse_ffnn, rsmse_DH, rsmse_DH_leland))
# print("|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")

# with open(errors_path_prefix + "Comparison_RSMSE_" + str(nbs_point_traj) + ".txt", "w") as rsmse_file:
#     # Writing data to a file
#     rsmse_file.write("|------------------------------------------------------Comparison of RSMSE------------------------------------------------------|\n")
#     rsmse_file.write("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\t\tDelta Hedge\t\t|\t\tLeland Delta Hedge\t\t|\n")
#     rsmse_file.write("|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     rsmse_file.write("|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(rsmse_trans, rsmse_gru, rsmse_lstm, rsmse_ffnn, rsmse_DH, rsmse_DH_leland))
#     rsmse_file.write("|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")

# print()

# smse_trans_trans = f.cdf(smse_trans/smse_trans, test_size-1, test_size-1)
# smse_trans_gru = f.cdf(smse_trans/smse_gru, test_size-1, test_size-1)
# smse_trans_lstm = f.cdf(smse_trans/smse_lstm, test_size-1, test_size-1)
# smse_trans_ffnn = f.cdf(smse_trans/smse_ffnn, test_size-1, test_size-1)
# smse_trans_DH = f.cdf(smse_trans/smse_DH, test_size-1, test_size-1)
# smse_trans_leland = f.cdf(smse_trans/smse_DH_leland, test_size-1, test_size-1)

# smse_gru_trans = f.cdf(smse_gru/smse_trans, test_size-1, test_size-1)
# smse_gru_gru = f.cdf(smse_gru/smse_gru, test_size-1, test_size-1)
# smse_gru_lstm = f.cdf(smse_gru/smse_lstm, test_size-1, test_size-1)
# smse_gru_ffnn = f.cdf(smse_gru/smse_ffnn, test_size-1, test_size-1)
# smse_gru_DH = f.cdf(smse_gru/smse_DH, test_size-1, test_size-1)
# smse_gru_leland = f.cdf(smse_gru/smse_DH_leland, test_size-1, test_size-1)

# smse_lstm_trans = f.cdf(smse_lstm/smse_trans, test_size-1, test_size-1)
# smse_lstm_gru = f.cdf(smse_lstm/smse_gru, test_size-1, test_size-1)
# smse_lstm_lstm = f.cdf(smse_lstm/smse_lstm, test_size-1, test_size-1)
# smse_lstm_ffnn = f.cdf(smse_lstm/smse_ffnn, test_size-1, test_size-1)
# smse_lstm_DH = f.cdf(smse_lstm/smse_DH, test_size-1, test_size-1)
# smse_lstm_leland = f.cdf(smse_lstm/smse_DH_leland, test_size-1, test_size-1)

# smse_ffnn_trans = f.cdf(smse_ffnn/smse_trans, test_size-1, test_size-1)
# smse_ffnn_gru = f.cdf(smse_ffnn/smse_gru, test_size-1, test_size-1)
# smse_ffnn_lstm = f.cdf(smse_ffnn/smse_lstm, test_size-1, test_size-1)
# smse_ffnn_ffnn = f.cdf(smse_ffnn/smse_ffnn, test_size-1, test_size-1)
# smse_ffnn_DH = f.cdf(smse_ffnn/smse_DH, test_size-1, test_size-1)
# smse_ffnn_leland = f.cdf(smse_ffnn/smse_DH_leland, test_size-1, test_size-1)

# smse_DH_trans = f.cdf(smse_DH/smse_trans, test_size-1, test_size-1)
# smse_DH_gru = f.cdf(smse_DH/smse_gru, test_size-1, test_size-1)
# smse_DH_lstm = f.cdf(smse_DH/smse_lstm, test_size-1, test_size-1)
# smse_DH_ffnn = f.cdf(smse_DH/smse_ffnn, test_size-1, test_size-1)
# smse_DH_DH = f.cdf(smse_DH/smse_DH, test_size-1, test_size-1)
# smse_DH_leland = f.cdf(smse_DH/smse_DH_leland, test_size-1, test_size-1)

# smse_leland_trans = f.cdf(smse_DH_leland/smse_trans, test_size-1, test_size-1)
# smse_leland_gru = f.cdf(smse_DH_leland/smse_gru, test_size-1, test_size-1)
# smse_leland_lstm = f.cdf(smse_DH_leland/smse_lstm, test_size-1, test_size-1)
# smse_leland_ffnn = f.cdf(smse_DH_leland/smse_ffnn, test_size-1, test_size-1)
# smse_leland_DH = f.cdf(smse_DH_leland/smse_DH, test_size-1, test_size-1)
# smse_leland_leland = f.cdf(smse_DH_leland/smse_DH_leland, test_size-1, test_size-1)

# print("|---------------------------------------------------------------F test for Smaller SMSE----------------------------------------------------------------|")
# print("|\t\t\t|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|\tDelta Hedge\t|\tLeland Delta Hedge\t|")
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tTransformer\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(smse_trans_trans, smse_trans_gru, smse_trans_lstm, smse_trans_ffnn, smse_trans_DH, smse_trans_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tGRU\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(smse_gru_trans, smse_gru_gru, smse_gru_lstm, smse_gru_ffnn, smse_gru_DH, smse_gru_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tLSTM\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(smse_lstm_trans, smse_lstm_gru, smse_lstm_lstm, smse_lstm_ffnn, smse_lstm_DH, smse_lstm_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tFFNN\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(smse_ffnn_trans, smse_ffnn_gru, smse_ffnn_lstm, smse_ffnn_ffnn, smse_ffnn_DH, smse_ffnn_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tDelta Hedge\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(smse_DH_trans, smse_DH_gru, smse_DH_lstm, smse_DH_ffnn, smse_DH_DH, smse_DH_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tLeland Delta Hedge\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(smse_leland_trans, smse_leland_gru, smse_leland_lstm, smse_leland_ffnn, smse_leland_DH, smse_leland_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")

# with open(errors_path_prefix + "SMSE_F_TEST_" + str(nbs_point_traj) + ".txt", "w") as smse_file:
#     # Writing data to a file
#     smse_file.write("|----------------------------------------------------------------F test for Smaller SMSE----------------------------------------------------------------|\n")
#     smse_file.write("|\t\t\t\t\t\t|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\t\tDelta Hedge\t\t|\t\tLeland Delta Hedge\t\t|\n")
#     smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     smse_file.write("|\t\tTransformer\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(smse_trans_trans, smse_trans_gru, smse_trans_lstm, smse_trans_ffnn, smse_trans_DH, smse_trans_leland))
#     smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     smse_file.write("|\t\tGRU\t\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(smse_gru_trans, smse_gru_gru, smse_gru_lstm, smse_gru_ffnn, smse_gru_DH, smse_gru_leland))
#     smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     smse_file.write("|\t\tLSTM\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(smse_lstm_trans, smse_lstm_gru, smse_lstm_lstm, smse_lstm_ffnn, smse_lstm_DH, smse_lstm_leland))
#     smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     smse_file.write("|\t\tFFNN\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(smse_ffnn_trans, smse_ffnn_gru, smse_ffnn_lstm, smse_ffnn_ffnn, smse_ffnn_DH, smse_ffnn_leland))
#     smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     smse_file.write("|\t\tDelta Hedge\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(smse_DH_trans, smse_DH_gru, smse_DH_lstm, smse_DH_ffnn, smse_DH_DH, smse_DH_leland))
#     smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     smse_file.write("|\tLeland Delta Hedge\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(smse_leland_trans, smse_leland_gru, smse_leland_lstm, smse_leland_ffnn, smse_leland_DH, smse_leland_leland))
#     smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")

# print()

# print("|----------------------------------------------------------------------------------------------Comparison of Mean Hedging Losses-----------------------------------------------------------------------------------------------|")
# print("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t\t|\t\tFFNN\t\t|\t\tDelta Hedge\t\t|\t\tLeland Delta Hedge\t\t|")
# print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|----------------------------------------------|")
# print("|\t{:.4f} +- {:.4f}\t\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t\t|\t{:.4f} +- {:.4f}\t\t|".format(np.mean(hedging_err_trans), np.std(hedging_err_trans, ddof=1), np.mean(hedging_err_gru), np.std(hedging_err_gru, ddof=1), np.mean(hedging_err_lstm), np.std(hedging_err_lstm, ddof=1), np.mean(hedging_err_ffnn), np.std(hedging_err_ffnn, ddof=1), np.mean(hedging_err_DH), np.std(hedging_err_DH, ddof=1), np.mean(hedging_err_DH_leland), np.std(hedging_err_DH_leland, ddof=1)))
# print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|----------------------------------------------|")

# with open(errors_path_prefix + "mean_hedging_error_" + str(nbs_point_traj) + ".txt", "w") as mean_file:
#     # Writing data to a file
#     mean_file.write("|----------------------------------------------------------------------------------------------Comparison of Mean Hedging Losses------------------------------------------------------------------------------------------------|\n")
#     mean_file.write("|\t\t\t\tTransformer\t\t\t\t|\t\t\tGRU\t\t\t\t\t|\t\t\tLSTM\t\t\t\t|\t\t\tFFNN\t\t\t\t|\t\t\t\tDelta Hedge\t\t\t\t|\t\t\t\tLeland Delta Hedge\t\t\t\t|\n")
#     mean_file.write("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|-----------------------------------------------|\n")
#     mean_file.write("|\t\t\t{:.4f} +- {:.4f}\t\t\t|\t\t{:.4f} +- {:.4f}\t\t|\t\t{:.4f} +- {:.4f}\t\t|\t\t{:.4f} +- {:.4f}\t\t|\t\t\t{:.4f} +- {:.4f}\t\t\t|\t\t\t{:.4f} +- {:.4f}\t\t\t\t\t|\n".format(np.mean(hedging_err_trans), np.std(hedging_err_trans, ddof=1), np.mean(hedging_err_gru), np.std(hedging_err_gru, ddof=1), np.mean(hedging_err_lstm), np.std(hedging_err_lstm, ddof=1), np.mean(hedging_err_ffnn), np.std(hedging_err_ffnn, ddof=1), np.mean(hedging_err_DH), np.std(hedging_err_DH, ddof=1), np.mean(hedging_err_DH_leland), np.std(hedging_err_DH_leland, ddof=1)))
#     mean_file.write("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|-----------------------------------------------|\n")

# print()

# mean_trans_trans = ttest_ind(hedging_err_trans, hedging_err_trans, equal_var=False, alternative="less").pvalue
# mean_trans_gru = ttest_ind(hedging_err_trans, hedging_err_gru, equal_var=False, alternative="less").pvalue
# mean_trans_lstm = ttest_ind(hedging_err_trans, hedging_err_lstm, equal_var=False, alternative="less").pvalue
# mean_trans_ffnn = ttest_ind(hedging_err_trans, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
# mean_trans_DH = ttest_ind(hedging_err_trans, hedging_err_DH, equal_var=False, alternative="less").pvalue
# mean_trans_leland = ttest_ind(hedging_err_trans, hedging_err_DH_leland, equal_var=False, alternative="less").pvalue

# mean_gru_trans = ttest_ind(hedging_err_gru, hedging_err_trans, equal_var=False, alternative="less").pvalue
# mean_gru_gru = ttest_ind(hedging_err_gru, hedging_err_gru, equal_var=False, alternative="less").pvalue
# mean_gru_lstm = ttest_ind(hedging_err_gru, hedging_err_lstm, equal_var=False, alternative="less").pvalue
# mean_gru_ffnn = ttest_ind(hedging_err_gru, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
# mean_gru_DH = ttest_ind(hedging_err_gru, hedging_err_DH, equal_var=False, alternative="less").pvalue
# mean_gru_leland = ttest_ind(hedging_err_gru, hedging_err_DH_leland, equal_var=False, alternative="less").pvalue

# mean_lstm_trans = ttest_ind(hedging_err_lstm, hedging_err_trans, equal_var=False, alternative="less").pvalue
# mean_lstm_gru = ttest_ind(hedging_err_lstm, hedging_err_gru, equal_var=False, alternative="less").pvalue
# mean_lstm_lstm = ttest_ind(hedging_err_lstm, hedging_err_lstm, equal_var=False, alternative="less").pvalue
# mean_lstm_ffnn = ttest_ind(hedging_err_lstm, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
# mean_lstm_DH = ttest_ind(hedging_err_lstm, hedging_err_DH, equal_var=False, alternative="less").pvalue
# mean_lstm_leland = ttest_ind(hedging_err_lstm, hedging_err_DH_leland, equal_var=False, alternative="less").pvalue

# mean_ffnn_trans = ttest_ind(hedging_err_ffnn, hedging_err_trans, equal_var=False, alternative="less").pvalue
# mean_ffnn_gru = ttest_ind(hedging_err_ffnn, hedging_err_gru, equal_var=False, alternative="less").pvalue
# mean_ffnn_lstm = ttest_ind(hedging_err_ffnn, hedging_err_lstm, equal_var=False, alternative="less").pvalue
# mean_ffnn_ffnn = ttest_ind(hedging_err_ffnn, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
# mean_ffnn_DH = ttest_ind(hedging_err_ffnn, hedging_err_DH, equal_var=False, alternative="less").pvalue
# mean_ffnn_leland = ttest_ind(hedging_err_ffnn, hedging_err_DH_leland, equal_var=False, alternative="less").pvalue

# mean_DH_trans = ttest_ind(hedging_err_DH, hedging_err_trans, equal_var=False, alternative="less").pvalue
# mean_DH_gru = ttest_ind(hedging_err_DH, hedging_err_gru, equal_var=False, alternative="less").pvalue
# mean_DH_lstm = ttest_ind(hedging_err_DH, hedging_err_lstm, equal_var=False, alternative="less").pvalue
# mean_DH_ffnn = ttest_ind(hedging_err_DH, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
# mean_DH_DH = ttest_ind(hedging_err_DH, hedging_err_DH, equal_var=False, alternative="less").pvalue
# mean_DH_leland = ttest_ind(hedging_err_DH, hedging_err_DH_leland, equal_var=False, alternative="less").pvalue

# mean_leland_trans = ttest_ind(hedging_err_DH_leland, hedging_err_trans, equal_var=False, alternative="less").pvalue
# mean_leland_gru = ttest_ind(hedging_err_DH_leland, hedging_err_gru, equal_var=False, alternative="less").pvalue
# mean_leland_lstm = ttest_ind(hedging_err_DH_leland, hedging_err_lstm, equal_var=False, alternative="less").pvalue
# mean_leland_ffnn = ttest_ind(hedging_err_DH_leland, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
# mean_leland_DH = ttest_ind(hedging_err_DH_leland, hedging_err_DH, equal_var=False, alternative="less").pvalue
# mean_leland_leland = ttest_ind(hedging_err_DH_leland, hedging_err_DH_leland, equal_var=False, alternative="less").pvalue

# print("|--------------------------------------------------------T-Test for Smaller Mean Hedging Losses--------------------------------------------------------|")
# print("|\t\t\t|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|\tDelta Hedge\t|\tLeland Delta Hedge\t|")
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tTransformer\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(mean_trans_trans, mean_trans_gru, mean_trans_lstm, mean_trans_ffnn, mean_trans_DH, mean_trans_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tGRU\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(mean_gru_trans, mean_gru_gru, mean_gru_lstm, mean_gru_ffnn, mean_gru_DH, mean_gru_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tLSTM\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(mean_lstm_trans, mean_lstm_gru, mean_lstm_lstm, mean_lstm_ffnn, mean_lstm_DH, mean_lstm_DH))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tFFNN\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(mean_ffnn_trans, mean_ffnn_gru, mean_ffnn_lstm, mean_ffnn_ffnn, mean_ffnn_DH, mean_ffnn_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tDelta Hedge\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(mean_DH_trans, mean_DH_gru, mean_DH_lstm, mean_DH_ffnn, mean_DH_DH, mean_DH_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")
# print("|\tLeland Delta Hedge\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|\t{:.4f}\t\t|".format(mean_leland_trans, mean_leland_gru, mean_leland_lstm, mean_leland_ffnn, mean_leland_DH, mean_leland_leland))
# print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|------------------------------|")

# with open(errors_path_prefix + "Mean_T_TEST_" + str(nbs_point_traj) + ".txt", "w") as mean_test_file:
#     # Writing data to a file
#     mean_test_file.write("|--------------------------------------------------------T-Test for Smaller Mean Hedging Losses---------------------------------------------------------|\n")
#     mean_test_file.write("|\t\t\t\t\t\t|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\t\tDelta Hedge\t\t|\t\tLeland Delta Hedge\t\t|\n")
#     mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     mean_test_file.write("|\t\tTransformer\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(mean_trans_trans, mean_trans_gru, mean_trans_lstm, mean_trans_ffnn, mean_trans_DH, mean_trans_leland))
#     mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     mean_test_file.write("|\t\tGRU\t\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(mean_gru_trans, mean_gru_gru, mean_gru_lstm, mean_gru_ffnn, mean_gru_DH, mean_gru_leland))
#     mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     mean_test_file.write("|\t\tLSTM\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(mean_lstm_trans, mean_lstm_gru, mean_lstm_lstm, mean_lstm_ffnn, mean_lstm_DH, mean_lstm_leland))
#     mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     mean_test_file.write("|\t\tFFNN\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(mean_ffnn_trans, mean_ffnn_gru, mean_ffnn_lstm, mean_ffnn_ffnn, mean_ffnn_DH, mean_ffnn_leland))
#     mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     mean_test_file.write("|\t\tDelta Hedge\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(mean_DH_trans, mean_DH_gru, mean_DH_lstm, mean_DH_ffnn, mean_DH_DH, mean_DH_leland))
#     mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")
#     mean_test_file.write("|\tLeland Delta Hedge\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t\t\t\t\t|\n".format(mean_leland_trans, mean_leland_gru, mean_leland_lstm, mean_leland_ffnn, mean_leland_DH, mean_leland_leland))
#     mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|-------------------------------|\n")

# print()

# def count_parameters(agent):
#     return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)

# print("TRANSFORMER PARAMETERS: ", count_parameters(agent_trans))
# print("GRU PARAMETERS: ", count_parameters(agent_gru))
# print("LSTM PARAMETERS: ", count_parameters(agent_lstm))
# print("FFNN PARAMETERS: ", count_parameters(agent))

# all_losses_fig = plt.figure(figsize=(10, 5))
# plt.plot(all_losses_lstm, label="LSTM")
# plt.plot(all_losses_ffnn, label="FFNN")
# plt.plot(all_losses_trans, label="Transformer")
# plt.plot(all_losses_gru, label="GRU")
# plt.xlabel("Training Iteration")
# plt.ylabel(loss_type)
# plt.legend()
# plt.title("Training " + loss_type + " per Iteration")
# plt.savefig(errors_path_prefix + "all_losses" + str(nbs_point_traj) + ".png")

# epoch_losses_fig = plt.figure(figsize=(10, 5))
# plt.plot(lstm_losses, label="LSTM")
# plt.plot(ffnn_losses, label="FFNN")
# plt.plot(trans_losses, label="Transformer")
# plt.plot(gru_losses, label="GRU")
# plt.xlabel('Epoch')
# plt.ylabel(loss_type)
# plt.legend()
# plt.title("Training " + loss_type + " per Epoch")
# plt.savefig(errors_path_prefix + "epoch_losses" + str(nbs_point_traj) + ".png")

# log_epoch_losses_fig = plt.figure(figsize=(10, 5))
# plt.plot(lstm_losses, label="LSTM")
# plt.plot(ffnn_losses, label="FFNN")
# plt.plot(trans_losses, label="Transformer")
# plt.plot(gru_losses, label="GRU")
# plt.yscale("log")
# plt.xlabel("Epoch")
# plt.ylabel("Log " + loss_type)
# plt.legend()
# plt.title("Trainging Log " + loss_type + " per Epoch")
# plt.savefig(errors_path_prefix + "log_epoch_losses" + str(nbs_point_traj) + ".png")

# fig = plt.figure(figsize=(10, 5))
# plt.hist([hedging_err_gru, hedging_err_trans], bins=50, label=["GRU", "Transformer"])
# plt.xlabel('Hedging losses')
# plt.ylabel('Frequency')
# plt.legend()
# plt.title("Hedging losses for GRU / Transformer - " + str(nbs_point_traj))
# plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_GRU_Transformer" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_gru, hedging_err_lstm], bins=50, label=["GRU", "LSTM"])
plt.xlabel('Hedging losses')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging losses for GRU / LSTM - " + str(nbs_point_traj))
plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_GRU_LSTM" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_gru, hedging_err_ffnn], bins=50, label=["GRU", "FFNN"])
plt.xlabel('Hedging losses')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging losses for GRU / FFNN - " + str(nbs_point_traj))
plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_GRU_FFNN" + str(nbs_point_traj) + ".png")

# fig = plt.figure(figsize=(10, 5))
# plt.hist([hedging_err_lstm, hedging_err_trans], bins=50, label=["LSTM", "Transformer"])
# plt.xlabel('Hedging losses')
# plt.ylabel('Frequency')
# plt.legend()
# plt.title("Hedging losses for LSTM / Transformer - " + str(nbs_point_traj))
# plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_LSTM_Transformer" + str(nbs_point_traj) + ".png")

# fig = plt.figure(figsize=(10, 5))
# plt.hist([hedging_err_ffnn, hedging_err_trans], bins=50, label=["ffnn", "Transformer"])
# plt.xlabel('Hedging losses')
# plt.ylabel('Frequency')
# plt.legend()
# plt.title("Hedging losses for FFNN / Transformer - " + str(nbs_point_traj))
# plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_FFNN_Transformer" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_ffnn, hedging_err_lstm], bins=50, label=["ffnn", "LSTM"])
plt.xlabel('Hedging losses')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging losses for FFNN / LSTM - " + str(nbs_point_traj))
plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_FFNN_LSTM" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_ffnn, hedging_err_DH], bins=50, label=["FFNN", "Delta-Hedge"])
plt.xlabel('Hedging losses')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging losses for FFNN vs Delta-Hedge - " + str(nbs_point_traj))
plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_FFNN_DH" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_DH_leland, hedging_err_DH], bins=50, label=["Leland Delta-Hedge", "Black-Scholes Delta-Hedge"])
plt.xlabel('Hedging losses')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging losses for Leland Delta-Hedge vs Black-Scholes Delta-Hedge - " + str(nbs_point_traj))
plt.savefig(errors_path_prefix + "hedging_errors/Hedging_Errors_Leland_DH" + str(nbs_point_traj) + ".png")

# Does not work with Transformers
# point_pred = agent.point_predict(t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)