import datetime as dt
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import Utils_general
import option_hedging.code_pytorch.DeepHedgingEnvironment as DeepHedgingEnvironment
import option_hedging.code_pytorch.old_models.DeepAgentLight as DeepAgentLight
import option_hedging.code_pytorch.old_models.DeepAgentMedium as DeepAgentMedium
import option_hedging.code_pytorch.old_models.DeepAgentTransformer as DeepAgentTransformer
import option_hedging.code_pytorch.old_models.DeepAgentLSTM as DeepAgentLSTM
import option_hedging.code_pytorch.old_models.DeepAgentGRU as DeepAgentGRU
from scipy.stats import ttest_ind
import pickle

batch_size = 256
train_size = 100000
test_size = 100000
epochs = 100
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.0, 0.1952]
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
generate_dataset = False

colours = ["g", "r", "m"]
font = {'size': 20}
matplotlib.rc('font', **font)

Ts = [1/252]
#Ts = [252/252, 30/252, 1/252]
for T in Ts:
    if T == 252/252:
        time_period = "year"
        nbs_point_traj = 13
        impact_values = [0.02, 0.01, 0.0]
    elif T == 30/252:
        time_period = "month"
        nbs_point_traj = 31
        impact_values = [0.02, 0.01, 0.0]
    else:
        time_period = "day"
        nbs_point_traj = 9
        impact_values = [0.001]

    if (option_type == 'Call'):
        V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
    else:
        V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

    N = nbs_point_traj - 1
    tick_interval = 1.0 if N < 10 else 2.0
    delta_t = T / N

    if generate_dataset:
        # Creating test dataset
        print("Generating Test Data Set")
        mu, sigma = params_vect
        test_set = S_0 * torch.ones(int(test_size/batch_size), nbs_point_traj, batch_size)
        for i in range(int(test_size/batch_size)):
            S_t = S_0 * torch.ones(batch_size)
            for j in range(N):
                Z = torch.randn(batch_size)
                S_t = S_t * torch.exp((mu - sigma ** 2 / 2) * delta_t + sigma * math.sqrt(delta_t) * Z)
                test_set[i, j+1, :] = S_t

        torch.save(test_set, "/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/test_set_single_path_{time_period}".format(time_period=time_period))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/test_set_single_path_{time_period}".format(time_period=time_period))
    test_set = test_set.to(device)

    strike_path = torch.ones(batch_size, nbs_point_traj, batch_size, device=device)*strike

    loss_type = "RSMSE"
    impact_persistence = [0, -math.log(0.5)/delta_t, -1] # impact persistence of {1, 0.5, 0}
    impact_persistence_strings = ["0", "-ln(0.5)", "infinity"]
    # impact_persistence = [-1]
    # impact_persistence_strings = ["infinity"]
    deltas_results = np.zeros((len(impact_persistence), len(impact_values), N))
    S_t_bids = np.zeros((len(impact_persistence), len(impact_values), nbs_point_traj))
    diff_deltas_results = np.zeros((len(impact_persistence), len(impact_values), N))
    
    deltas_DH_leland = np.zeros((len(impact_persistence), len(impact_values), N, 1))
    hedging_err_DH_leland = np.zeros((len(impact_persistence), len(impact_values), N, 1))
    diff_deltas_DH_leland = np.zeros((len(impact_persistence), len(impact_values), N, 1))

    deltas_results_strike_path = np.zeros((len(impact_persistence), len(impact_values), N))
    S_t_bids_strike_path = np.zeros((len(impact_persistence), len(impact_values), nbs_point_traj))
    diff_deltas_results_strike_path = np.zeros((len(impact_persistence), len(impact_values), N))

    deltas_DH_leland_strike_path = np.zeros((len(impact_persistence), len(impact_values), N, 1))
    hedging_err_DH_leland_strike_path = np.zeros((len(impact_persistence), len(impact_values), N, 1))
    diff_deltas_DH_leland_strike_path = np.zeros((len(impact_persistence), len(impact_values), N, 1))

    for p, persistence in enumerate(impact_persistence):
        lambdas = [persistence, persistence]
        for i, impact in enumerate(impact_values):
            alpha = 1.0+impact
            beta = 1.0-impact
            
            name = "code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/{loss_type}_hedging_{impact:.4f}_market_impact_single_path_{time_period}_{persistence:.1f}_persistence".format(time_period_folder=time_period, loss_type="quadratic" if loss_type=="RMSE" else "semi_quadratic", impact=impact, time_period=time_period, persistence=persistence)
            if light:
                agent = DeepAgentLight.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                                nbs_shares, lambdas, name=name)
            else:
                agent = DeepHedgingEnvironment.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                                loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, dropout, prepro_stock,
                                nbs_shares, lambdas, name=name)

            print("START FFNN {loss_type} - {impact:.4f} IMPACT - {persistence:.1f} Persistence".format(loss_type=loss_type, impact=impact, persistence=persistence))
            all_losses, epoch_losses = agent.train(train_size = train_size, epochs=epochs, lr_schedule=lr_schedule)

            print("DONE FFNN {loss_type} - {impact:.4f} IMPACT - {persistence:.1f} Persistence".format(loss_type=loss_type, impact=impact, persistence=persistence))
            agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/{name}".format(name=name))
            deltas, hedging_err, S_t, V_t, A_t, B_t, = agent.test(test_size=test_size, test_set=test_set)
            deltas_strike_path, hedging_err_strike_path, S_t_strike_path, V_t_strike_path, A_t_strike_path, B_t_strike_path = agent.test(test_size=batch_size, test_set=strike_path)
            
            if generate_dataset:
                if p == 0 and i == 0:
                    max_error_index = np.argmax(hedging_err)
                with open('single_path_index_{time_period}.pickle'.format(time_period=time_period), 'wb') as single_path_index:
                    pickle.dump(max_error_index, single_path_index)
            else:
                with open('single_path_index_{time_period}.pickle'.format(time_period=time_period), 'rb') as single_path_index:
                    max_error_index = pickle.load(single_path_index)
                    print("MAX ERROR INDEX: ", max_error_index)
            
            deltas_results[p, i] = deltas[:,max_error_index,0]
            deltas_results_strike_path[p, i] = deltas_strike_path[:,0,0]
            
            for k in range(N):
                if k == 0:
                    diff_deltas_results[p, i, k] = 0
                    diff_deltas_results_strike_path[p, i, k] = 0
                else:
                    diff_deltas_results[p, i, k] = deltas_results[p, i, k].item() - deltas_results[p, i, k-1].item()
                    diff_deltas_results_strike_path[p, i, k] = deltas_results_strike_path[p, i, k].item() - deltas_results_strike_path[p, i, k-1].item()
            
            # sells = np.where(diff_deltas_results < 0, -diff_deltas_results, 0)
            # sells = np.concatenate((sells[:, :, :], np.ones((sells.shape[0], sells.shape[1], 1))), axis=2)

            # print("deltas.shape: ", deltas.shape)
            # print("deltas_results[p, i].shape: ", deltas_results[p, i].shape)
            # print("deltas_results[p, i]: ", deltas_results[p, i])
            # print("sells.shape: ", sells.shape)
            # print("sells[p, i]: ", sells[p, i])
            # print("diff_deltas_results[p, i].shape: ", diff_deltas_results[p, i].shape)
            # print("diff_deltas_results[p, i]: ", diff_deltas_results[p, i])
            # print("B_t[:, max_error_index].shape: ", B_t[:, max_error_index].shape)
            # print("B_t[:, max_error_index]: ", B_t[:, max_error_index])

            S_t_bids[p, i] = S_t[:, max_error_index] * ((1 + 1 + B_t[:, max_error_index]) ** beta - (1 + B_t[:, max_error_index]) ** beta)
            S_t_bids_strike_path[p, i] = S_t_strike_path[:, 0] * ((1 + 1 + B_t_strike_path[:, 0]) ** beta - (1 + B_t_strike_path[:, 0]) ** beta)
            
            # print("S_t_bids[p, i].shape: ", S_t_bids[p, i].shape)
            # print("S_t_bids[p, i]: ", S_t_bids[p, i])

            semi_square_hedging_err = np.square(np.where(hedging_err > 0, hedging_err, 0))
            rsmse = np.sqrt(np.mean(semi_square_hedging_err))

            print(" ----------------- ")
            print(" Deep Hedging {loss_type} FFNN Results".format(loss_type=loss_type))
            print(" ----------------- ")
            Utils_general.print_stats(hedging_err, deltas, loss_type, "Deep hedge - FFNN - {}".format(loss_type), V_0)

            def count_parameters(agent):
                return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
            
            deltas_DH_leland[p, i], hedging_err_DH_leland[p, i] = Utils_general.delta_hedge_res(S_t[:,max_error_index:max_error_index+1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas, Leland=True)
            deltas_DH_leland_strike_path[p, i], hedging_err_DH_leland_strike_path[p, i] = Utils_general.delta_hedge_res(S_t_strike_path[:,0:1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas, Leland=True)

            for k in range(N):
                if k == 0:
                    diff_deltas_DH_leland[p, i, k] = 0
                    diff_deltas_DH_leland_strike_path[p, i, k] = 0
                else:
                    diff_deltas_DH_leland[p, i, k] = deltas_DH_leland[p, i, k, :] - deltas_DH_leland[p, i, k-1, :]
                    diff_deltas_DH_leland_strike_path[p, i, k] = deltas_DH_leland_strike_path[p, i, k, :] - deltas_DH_leland_strike_path[p, i, k-1, :]


        deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t[:,max_error_index:max_error_index+1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
        deltas_DH_strike_path, hedging_err_DH_strike_path = Utils_general.delta_hedge_res(S_t_strike_path[:,0:1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)

        fig = plt.figure(figsize=(15, 10))
        plt.plot(S_t[:,max_error_index], label="Underlying asset prices sequence", color="k")
        # for j in range(len(impact_values)):
        #     if impact_values[j] != 0:
        #         plt.plot(S_t_bids[p, j, :], label="Unit prices path with beta = {:.4f}".format(1.0-impact_values[j]), color=colours[j])
        plt.xlabel(r'Time steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Underlying asset prices ($S_t$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of stock prices - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f} - lambdas = {persistence:.1f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1], persistence=persistence))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/path_of_stock_prices_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        fig = plt.figure(figsize=(15, 10))
        plt.plot(deltas_DH, label="Black-Scholes delta hedge baseline")
        plt.plot(deltas_DH_leland[p, 0, :], label="Leland delta hedge baseline with {impact}".format(impact="alpha = {:.3f}, beta = {:.3f}".format(1.0+impact_values[0], 1.0-impact_values[0])))
        for j in range(len(impact_values)):
            plt.plot(deltas_results[p, j, :], label="Optimal DRL hedge with {impact}".format(impact="no impact" if impact_values[j] == 0.0 and impact_values[j] == 0.0 else "alpha = {:.3f}, beta = {:.3f}".format(1.0+impact_values[j], 1.0-impact_values[j])))
        plt.xlabel(r'Time steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Hedging positions ($X_{t+1}$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of deltas - Time Period: {time_period} - only RSMSE - shares = 1 - lambdas = {persistence:.1f}".format(time_period=time_period, persistence=persistence))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/path_of_deltas_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        diff_deltas_DH = np.ones((deltas_DH.shape[0], deltas_DH.shape[1]))
        for j in range(deltas_DH.shape[0]):
            if j == 0:
                diff_deltas_DH[j] = 0
            else:
                diff_deltas_DH[j] = deltas_DH[j] - deltas_DH[j-1]

        # fig = plt.figure(figsize=(15, 10))
        # plt.plot(diff_deltas_DH, label="Black-Scholes delta hedge")
        # plt.plot(diff_deltas_DH_leland[p, 0], label="Leland delta hedge - {impact}".format(impact="alpha = {:.4f}, beta = {:.4f}".format(1.0+impact_values[0], 1.0-impact_values[0])))
        # for j in range(len(impact_values)):
        #     plt.plot(diff_deltas_results[p, j], label="Optimal hedge - {impact}".format(impact="no impact" if impact_values[j] == 0.0 and impact_values[j] == 0.0 else "alpha = {:.4f}, beta = {:.4f}".format(1.0+impact_values[j], 1.0-impact_values[j])))
        # plt.xlabel('Time steps (t)', fontsize=22)
        # plt.ylabel('Difference share of stock (diff X_{t+1})', fontsize=22)
        # plt.grid()
        # plt.legend()
        # # plt.title("Path of diff deltas - Time Period: {time_period} - only RSMSE - shares = 1 - lambdas = {persistence:.1f}".format(time_period=time_period, persistence=persistence))
        # plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/path_of_diff_deltas_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        print()

        #For the path having the same spot prices as the strike prices:
        fig = plt.figure(figsize=(15, 10))
        plt.plot(S_t_strike_path[:,0], label="Underlying asset prices sequence", color="k")
        # for j in range(len(impact_values)):
        #     if impact_values[j] != 0:
        #         plt.plot(S_t_bids_strike_path[p, j, :], label="Unit prices path with beta = {:.4f}".format(1.0-impact_values[j],), color=colours[j])
        plt.xlabel(r'Time Steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Underlying asset prices ($S_t$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of stock prices equal to the strike price - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f} - lambdas = {persistence:.1f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1], persistence=persistence))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/strike_path/path_of_stock_prices_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        fig = plt.figure(figsize=(15, 10))
        plt.plot(deltas_DH_strike_path, label="Black-Scholes delta hedge baseline")
        plt.plot(deltas_DH_leland_strike_path[p, 0, :], label="Leland delta hedge baseline with {impact}".format(impact="alpha = {:.3f}, beta = {:.3f}".format(1.0+impact_values[0], 1.0-impact_values[0])))
        for j in range(len(impact_values)):
            plt.plot(deltas_results_strike_path[p, j, :], label="Optimal DRL hedge with {impact}".format(impact="no impact" if impact_values[j] == 0.0 and impact_values[j] == 0.0 else "alpha = {:.3f}, beta = {:.3f}".format(1.0+impact_values[j], 1.0-impact_values[j])))
        plt.ylim([0.44, 0.59])
        plt.xlabel(r'Time steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Hedging positions ($X_{t+1}$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of deltas with strike path - Time Period: {time_period} - only RSMSE - shares = 1 - lambdas = {persistence:.1f}".format(time_period=time_period, persistence=persistence))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/strike_path/path_of_deltas_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        diff_deltas_DH_strike_path = np.ones((deltas_DH_strike_path.shape[0], deltas_DH_strike_path.shape[1]))
        for j in range(deltas_DH_strike_path.shape[0]):
            if j == 0:
                diff_deltas_DH_strike_path[j] = 0
            else:
                diff_deltas_DH_strike_path[j] = deltas_DH_strike_path[j] - deltas_DH_strike_path[j-1]

        # fig = plt.figure(figsize=(15, 10))
        # plt.plot(diff_deltas_DH_strike_path, label="Delta hedge")
        # plt.plot(diff_deltas_DH_leland_strike_path[p, 0], label="Leland delta hedge - {impact}".format(impact="alpha = {:.4f}, beta = {:.4f}".format(1.0+impact_values[0], 1.0-impact_values[0])))
        # for j in range(len(impact_values)):
        #     plt.plot(diff_deltas_results_strike_path[p, j], label="Global {loss} - {impact}".format(loss=loss_type, impact="no impact" if impact_values[j] == 0.0 and impact_values[j] == 0.0 else "alpha = {:.4f}, beta = {:.4f}".format(1.0+impact_values[j], 1.0-impact_values[j])))
        # plt.xlabel('Time Steps', fontsize=22)
        # plt.ylabel('Difference share of stock (diff delta_{t+1})', fontsize=22)
        # plt.grid()
        # plt.legend()
        # # plt.title("Path of diff deltas with strike path - Time Period: {time_period} - only RSMSE - shares = 1 - lambdas = {persistence:.1f}".format(time_period=time_period, persistence=persistence))
        # plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/strike_path/path_of_diff_deltas_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        print()

    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact
        fig = plt.figure(figsize=(15, 10))
        plt.plot(S_t[:,max_error_index], label="Underlying asset prices sequence", color="k")
        # for p, persistence in enumerate(impact_persistence):
        #     plt.plot(S_t_bids[p, i, :], label="Unit prices path with lambdas = {persistence}".format(persistence=impact_persistence_strings[p]), color=colours[p])
        plt.xlabel(r'Time steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Underlying asset prices ($S_t$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of stock prices - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f} - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1], alpha=alpha, beta=beta))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/path_of_stock_prices_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))
    
    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact
        fig = plt.figure(figsize=(15, 10))
        plt.plot(deltas_DH, label="Black-Scholes delta hedge baseline")
        plt.plot(deltas_DH_leland[0, i, :], label="Leland delta hedge baseline with lambdas = {persistence}".format(persistence=impact_persistence_strings[p]))
        for p, persistence in enumerate(impact_persistence):
            plt.plot(deltas_results[p, i, :], label="Optimal DRL hedge with lambdas = {persistence}".format(persistence=impact_persistence_strings[p]))
        plt.xlabel(r'Time steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Hedging positions ($X_{t+1}$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of deltas - Time Period: {time_period} - only RSMSE - shares = 1 - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, alpha=alpha, beta=beta))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/path_of_deltas_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))

    # for i, impact in enumerate(impact_values):
    #     alpha = 1.0+impact
    #     beta = 1.0-impact
    #     fig = plt.figure(figsize=(15, 10))
    #     plt.plot(diff_deltas_DH, label="Black-Scholes delta hedge")
    #     plt.plot(diff_deltas_DH_leland[0, i], label="Leland Delta hedge - lambdas = {persistence:.1f}".format(persistence=impact_persistence[0]))
    #     for p, persistence in enumerate(impact_persistence):
    #         plt.plot(diff_deltas_results[p, i], label="Optimal hedge - lambdas = {persistence:.1f}".format(persistence=impact_persistence_strings[p]))
    #     plt.xlabel('Time Steps', fontsize=22)
    #     plt.ylabel('Difference share of stock (diff delta_{t+1})', fontsize=22)
    #     plt.grid()
    #     plt.legend()
    #     # plt.title("Path of diff deltas - Time Period: {time_period} - only RSMSE - shares = 1 - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, alpha=alpha, beta=beta))
    #     plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/path_of_diff_deltas_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))

    
    #For the path having the same spot prices as the strike prices:

    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact
        fig = plt.figure(figsize=(15, 10))
        plt.plot(S_t_strike_path[:,0], label="Underlying asset prices sequence", color="k")
        # for p, persistence in enumerate(impact_persistence):
        #     plt.plot(S_t_bids_strike_path[p, i, :], label="Unit prices path with lambdas = {persistence}".format(persistence=impact_persistence_strings[p]), color=colours[p])
        plt.xlabel(r'Time steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Underlying asset prices ($S_t$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of stock prices equal to the strike price - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f} - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1], alpha=alpha, beta=beta))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/strike_path/path_of_stock_prices_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))
    
    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact
        fig = plt.figure(figsize=(15, 10))
        plt.plot(deltas_DH_strike_path, label="Black-Scholes delta hedge baseline")
        plt.plot(deltas_DH_leland_strike_path[0, i, :], label="Leland delta hedge baseline with lambdas = {persistence}".format(persistence=impact_persistence_strings[p]))
        for p, persistence in enumerate(impact_persistence):
            plt.plot(deltas_results_strike_path[p, i, :], label="Optimal DRL hedge with lambdas = {persistence}".format(persistence=impact_persistence_strings[p]))
        plt.ylim([0.44, 0.59])
        plt.xlabel(r'Time steps ($t$)', fontsize=22)
        plt.xticks(np.arange(0, nbs_point_traj, tick_interval))
        plt.ylabel(r'Hedging positions ($X_{t+1}$)', fontsize=22)
        plt.grid()
        plt.legend()
        # plt.title("Path of deltas with strike path - Time Period: {time_period} - only RSMSE - shares = 1 - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, alpha=alpha, beta=beta))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/strike_path/path_of_deltas_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))

    # for i, impact in enumerate(impact_values):
    #     alpha = 1.0+impact
    #     beta = 1.0-impact
    #     fig = plt.figure(figsize=(15, 10))
    #     plt.plot(diff_deltas_DH_strike_path, label="Delta hedge")
    #     plt.plot(diff_deltas_DH_leland_strike_path[0, i], label="Leland delta hedge - lambdas = {persistence:.1f}".format(persistence=impact_persistence[0]))
    #     for p, persistence in enumerate(impact_persistence):
    #         plt.plot(diff_deltas_results_strike_path[p, i], label="Global {loss} - lambdas = {persistence:.1f}".format(loss=loss_type, persistence=persistence))
    #     plt.xlabel('Time Steps', fontsize=22)
    #     plt.ylabel('Difference share of stock (diff delta_{t+1})', fontsize=22)
    #     plt.grid()
    #     plt.legend()
    #     # plt.title("Path of diff deltas with strike path - Time Period: {time_period} - only RSMSE - shares = 1 - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, alpha=alpha, beta=beta))
    #     plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/strike_path/path_of_diff_deltas_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))
