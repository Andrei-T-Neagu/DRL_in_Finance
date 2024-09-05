import datetime as dt
import math
import sys
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import Utils_general
import DeepAgent as DeepAgent
import DeepAgentTransformer
import DeepAgentLSTM as DeepAgentLSTM
import DeepAgentGRU as DeepAgentGRU
from scipy.stats import ttest_ind

T = 252/252
batch_size = 256
train_size = 100000
test_size = 100000
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
    impact_values = [0.01]
elif T == 30/252:
    time_period = "month"
    nbs_point_traj = 31
    impact_values = [0.0, 0.01]
else:
    time_period = "day"
    nbs_point_traj = 9
    impact_values = [0.0, 0.001]

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

loss_types = ["RSMSE"]
for loss in loss_types:
    for impact in impact_values:
        loss_type = loss
        alpha = 1.0+impact
        beta = 1.0-impact

        name = "code_pytorch/effect_of_market_impact/{time_period}/{loss_type}_hedging_{impact}_market_impact".format(time_period=time_period, loss_type="quadratic" if loss_type=="RMSE" else "semi_quadratic", impact="no" if alpha==1.0 and beta==1.0 else "with")
        agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                        loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                        nbs_shares, lambdas, name=name)

        print("START FFNN {loss_type} {impact} IMPACT".format(loss_type=loss_type, impact="NO" if alpha==1.0 and beta==1.0 else "WITH"))
        all_losses, epoch_losses = agent.train(train_size=train_size, epochs=epochs, lr_schedule=lr_schedule)

        print("DONE FFNN {loss_type} {impact} IMPACT".format(loss_type=loss_type, impact="NO" if alpha==1.0 and beta==1.0 else "WITH"))
        agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name)
        deltas, hedging_err, S_t, V_t, A_t, B_t, = agent.test(test_size=test_size, test_set=test_set)
        semi_square_hedging_err = np.square(np.where(hedging_err > 0, hedging_err, 0))
        rsmse = np.sqrt(np.mean(semi_square_hedging_err))

        print(" ----------------- ")
        print(" Deep Hedging %s FFNN Results" % (loss_type))
        print(" ----------------- ")
        Utils_general.print_stats(hedging_err, deltas, loss_type, "Deep hedge - FFNN - %s" % (loss_type), V_0)

        def count_parameters(agent):
            return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)

        print("FFNN PARAMETERS: ", count_parameters(agent))

        # Single point prediction
        portfolio_values = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0]
        colours = ["green", "red", "purple", "brown", "pink", "gray"]
        spot_prices = np.linspace(700, 1300, num=25).tolist()
        spot_prices_dh = np.tile(spot_prices, (nbs_point_traj, 1))
        deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(spot_prices_dh, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas, Leland=False)
        deltas_DH_leland, hedging_err_DH_leland = Utils_general.delta_hedge_res(spot_prices_dh, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas, Leland=True)
        deltas = np.zeros((6, 25))
        for i, Vt in enumerate(portfolio_values):
            for j, spot in enumerate(spot_prices):
                point_pred = agent.point_predict(t=N/2, S_t=spot, V_t=Vt, A_t=0, B_t=0, delta_t=0.5)
                deltas[i, j] = point_pred
        
        font = {'size': 20}
        matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(15, 10))
        plt.plot(deltas_DH[int(N/2)], label="Black-Scholes delta hedge baseline", color="blue")
        if impact != 0:
            plt.plot(deltas_DH_leland[int(N/2)], label="Leland delta hedge baseline", color="orange")
        for i, Vt in enumerate(portfolio_values):
            plt.plot(deltas[i], label="Optimal DRL hedge with V_t = {:.1f}".format(int(Vt)), color=colours[i])
        prices = np.linspace(700, 1300, num=7).tolist()
        indices = list(range(0, len(spot_prices), 4))
        plt.ylim([-0.2, 1.9])
        plt.xlabel(r'Underlying asset prices ($S_t$)', fontsize=22)
        plt.xticks(indices, prices)
        plt.ylabel(r'Hedging positions ($X_{t+1}$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("ATM Call - Delta Hedge vs {loss_type} - {impact} - time_t = 0.5000".format(loss_type=loss_type, impact="No liquidity impact" if alpha==1.0 and beta==1.0 else "alpha = {:.4f} beta = {:.4f}").format(alpha, beta))
        plt.savefig("code_pytorch/effect_of_market_impact/{time_period}/{loss_type} hedging, {impact} market impact".format(time_period=time_period, loss_type="Quadratic" if loss_type=="RMSE" else "Semi-quadratic", impact="no" if alpha==1.0 and beta==1.0 else "with"))

        print()