import datetime as dt
import math
import sys
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Utils_general
import DeepAgent as DeepAgent
import DeepAgentTransformer as DeepAgentTransformer
import DeepAgentLSTM as DeepAgentLSTM
import DeepAgentGRU as DeepAgentGRU
from scipy.stats import ttest_ind

T = 252/252
batch_size = 256
train_size = 100000
test_size = 1000000
epochs = 100
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.0892, 0.1952]
S_0 = 1000
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 4
nbs_units = 256
lr = 0.0001
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [-1, -1]
dropout = 0

light = False
lr_schedule = True

if T == 252/252:
    time_period = "year"
    nbs_point_traj = 13
    impact_values = [0.0, 0.01, 0.02]
elif T == 30/252:
    time_period = "month"
    nbs_point_traj = 31
    impact_values = [0.0, 0.01, 0.02]
else:
    time_period = "day"
    nbs_point_traj = 9
    impact_values = [0.0, 0.001, 0.002]

if (option_type == 'call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

# Creating test dataset
mu, sigma = params_vect
N = nbs_point_traj - 1
delta_t = T / N
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_set = S_0 * torch.ones(int(test_size/batch_size), nbs_point_traj, batch_size, device=device)
for i in range(int(test_size/batch_size)):
    S_t = S_0 * torch.ones(batch_size, device=device)
    for j in range(N):
        Z = torch.randn(batch_size, device=device)
        S_t = S_t * torch.exp((mu - sigma ** 2 / 2) * delta_t + sigma * math.sqrt(delta_t) * Z)
        test_set[i, j+1, :] = S_t

test_set_DH = np.concatenate(test_set.detach().cpu().numpy(), axis=1)

impact_persistence = [0, -math.log(0.5)/delta_t, -1] # impact persistence of {1, 0.5, 0}

with open("code_pytorch/FFNN_performance_" + time_period + ".txt", 'w'): pass

for persistence in impact_persistence:
    for impact in impact_values:
        lambdas = [persistence, persistence]
        alpha = 1.0+impact
        beta = 1.0-impact

        name = "code_pytorch/effect_of_market_impact/{time_period}/{loss_type}_hedging_{impact}_market_impact_{nbs_layers:d}_layers".format(time_period=time_period, loss_type="quadratic" if loss_type=="RMSE" else "semi_quadratic", impact="no" if alpha==1.0 and beta==1.0 else "with", nbs_layers=nbs_layers)
        agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                        loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                        nbs_shares, lambdas, name=name)

        print("START FFNN alpha = {:.4f}, beta = {:.4f}, lamba_a = {:.2f}, lambda_b = {:.2f}".format(alpha, beta, lambdas[0], lambdas[1]))
        all_losses, epoch_losses = agent.train(train_size=train_size, epochs=epochs, lr_schedule=lr_schedule)

        print("DONE FFNN alpha = {:.4f}, beta = {:.4f}, lamba_a = {:.2f}, lambda_b = {:.2f}".format(alpha, beta, lambdas[0], lambdas[1]))
        agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name)
        deltas, hedging_err, S_t, V_t, A_t, B_t, = agent.test(test_size=test_size, test_set=test_set)

        with open("code_pytorch/FFNN_performance_" + time_period + ".txt", "a") as performance_file:
            
            sys.stdout = performance_file
            print("####################################################################")
            print("alpha = {:.4f}, beta = {:.4f}, lamba_a = {:.2f}, lambda_b = {:.2f}".format(alpha, beta, lambdas[0], lambdas[1]))
            print("####################################################################")

            print(" ----------------- ")
            print(" Deep Hedging %s FFNN Results" % (loss_type))
            print(" ----------------- ")
            Utils_general.print_stats(hedging_err, deltas, loss_type, "Deep hedge - FFNN - %s" % (loss_type), V_0)
            
            print(" ----------------- ")
            print(" Delta Hedging Results")
            print(" ----------------- ")
            deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(test_set_DH, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
            Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)

            print(" ----------------- ")
            print("Leland Delta Hedging Results")
            print(" ----------------- ")
            deltas_DH_leland, hedging_err_DH_leland = Utils_general.delta_hedge_res(test_set_DH, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas, Leland=True)
            Utils_general.print_stats(hedging_err_DH_leland, deltas_DH_leland, "Leland delta hedge", "Leland delta hedge", V_0)
        
        sys.stdout = sys.__stdout__