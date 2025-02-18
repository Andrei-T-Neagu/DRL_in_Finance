import os
import datetime as datetime
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
from ray import tune, train
import ray
from ray.train import Checkpoint
import matplotlib.pyplot as plt
import Utils_general
import DeepHedgingEnvironment
import RL_algorithms.DQN as DQN
import RL_algorithms.PG as PG
import RL_algorithms.PPO as PPO
import RL_algorithms.DDPG as DDPG
from data_generation_processes.GARCH import GARCH
import yfinance as yf
import pickle
import tempfile
import shutil
import subprocess
import copy
from scipy.stats import ttest_ind

start_total_time = datetime.datetime.now()
episodes = 200000
trans_costs = 0.00              #proportional transaction costs 0.0 or 0.01
twin_delayed=True
double=False
dueling=False
T = 252/252

cpu = True
num_cpus = 60
cpus = 1
num_gpus = 1
gpus = 0.05
subprocess.Popen("nvidia-smi")

global_path_prefix = os.getcwd()+"/"

print("GLOBAL PATH PREFIX: ", global_path_prefix)

if T == 252/252:
    time_frame = "year"
    nbs_point_traj = 13
    start="2000-11-15"
    end="2024-10-15"
    interval= "1mo"             # Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
elif T == 30/252:
    time_frame = "month"
    start="2000-11-15"
    end="2024-10-15"
    interval= "1wk"
    nbs_point_traj = 5
elif T == 1/252:
    time_frame = "day"
    start=(datetime.date.today()-datetime.timedelta(days=720)).strftime("%Y-%m-%d")
    end=(datetime.date.today()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    interval= "1h"
    nbs_point_traj = 8

if twin_delayed:
    ddpg_model_type = "twin_delayed"
else:
    ddpg_model_type = "vanilla"

if dueling and double:
    dqn_model_type = "dueling_double"
elif dueling:
    dqn_model_type = "dueling"
elif double:
    dqn_model_type = "double"
else:
    dqn_model_type = "vanilla"

train_size = 2**19
val_size = 2**17
test_size = 2**17

num_test_sets = 10

r_borrow = 0
r_lend = 0

S_0 = 100
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 100

prepro_stock = "log-moneyness"
nbs_shares = 1

#real market data parameters for garch model
stock = "^GSPC"
garch_type="gjr"

# neural network type parameters
light = False
lr_schedule = True

state_size = 2 if light else 3

# Black-Scholes mu and sigma parameters estimated from real market data
# market_data = yf.download(stock, start=start, end=end, interval=interval, timeout=60)

# with open("market_data_" + stock + "_" + time_frame + ".pickle", 'wb') as file:
#     pickle.dump(market_data, file)

with open("market_data_" + stock + "_" + time_frame + ".pickle", "rb") as file:
    market_data = pickle.load(file)

with open("BS_market_data_" + stock + ".pickle", "rb") as file:
    BS_market_data = pickle.load(file)
log_returns = np.log(BS_market_data['Close'] / BS_market_data['Close'].shift(1)).dropna()
mu = log_returns.mean() * 252
sigma = log_returns.std() * np.sqrt(252)
params_vect = [mu, sigma]

# For reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initializing the inital option price using black-scholes option pricing
if (option_type == 'call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

# Initialize the garch model
garch_model = GARCH(stock=stock, market_data=market_data ,start=start, end=end, interval=interval, type=garch_type)

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

    torch.save(dataset, global_path_prefix + "option_hedging/BS_" + str(dataset_type))

# Training the GARCH model
def train_garch():
    params = garch_model.train(save_params=True)

# Generating the GARCH datasets
def generate_garch_dataset(dataset_type="train_set", size=train_size):
    print("Generating GARCH Data Set")
    if dataset_type == "test_set":
        test_dataset = np.ones((num_test_sets,size,nbs_point_traj))
        for i in range(num_test_sets):
            dataset = garch_model.generate(S_0=S_0, batch_size=size, num_points=nbs_point_traj, load_params=True)
            test_dataset[i] = dataset
        test_dataset = torch.from_numpy(test_dataset).to(torch.float)
        torch.save(test_dataset, global_path_prefix  + "option_hedging/" + str(dataset_type))    
    else:
        dataset = garch_model.generate(S_0=S_0, batch_size=size, num_points=nbs_point_traj, load_params=True)
        dataset = torch.from_numpy(dataset).to(torch.float)
        torch.save(dataset, global_path_prefix  + "option_hedging/" + str(dataset_type))

"""Training the garch model and generating the datasets"""
train_garch()
garch_model.print_params()
generate_garch_dataset(dataset_type="train_set", size=train_size)
generate_garch_dataset(dataset_type="val_set", size=val_size)
generate_garch_dataset(dataset_type="test_set", size=test_size)

# Select the device
if cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the training and testing datasets
train_set = torch.load(global_path_prefix + "option_hedging/train_set", weights_only=True)
val_set = torch.load(global_path_prefix + "option_hedging/val_set", weights_only=True)
test_set = torch.load(global_path_prefix + "option_hedging/test_set", weights_only=True)

# tune_test_set = copy.deepcopy(test_set[0])
# print("test_set.shape: ", test_set.shape)
# print("tune_test_set.shape: ", tune_test_set.shape)

# For reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

"""Print baseline models performance statistics"""
bs_rsmse_list = []
leland_rsmse_list = []
for i in range(num_test_sets):
    test_set_BS = test_set[i].detach().cpu().numpy().T

    print(" ----------------- ")
    print(" Delta Hedging Results")
    print(" ----------------- ")
    deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, 
                                                            nbs_shares=nbs_shares, trans_costs=trans_costs)
    Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)
    semi_square_hedging_err_DH = np.square(np.where(hedging_err_DH > 0, hedging_err_DH, 0))
    smse_DH = np.mean(semi_square_hedging_err_DH)
    rsmse_DH = np.sqrt(np.mean(semi_square_hedging_err_DH))
    print(rsmse_DH)
    bs_rsmse_list.append(rsmse_DH.item())

    print(" ----------------- ")
    print("Leland Delta Hedging Results")
    print(" ----------------- ")
    deltas_DH_leland, hedging_err_DH_leland = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0,
                                                                            nbs_shares=nbs_shares, trans_costs=trans_costs, Leland=True)
    Utils_general.print_stats(hedging_err_DH_leland, deltas_DH_leland, "Leland delta hedge", "Leland delta hedge", V_0)
    semi_square_hedging_err_DH_leland = np.square(np.where(hedging_err_DH_leland > 0, hedging_err_DH_leland, 0))
    smse_DH_leland = np.mean(semi_square_hedging_err_DH_leland)
    rsmse_leland = np.sqrt(np.mean(semi_square_hedging_err_DH_leland))
    leland_rsmse_list.append(rsmse_leland.item())
    if i == 0:
        bs_actions = deltas_DH
        rsmse_DH_leland = rsmse_leland
        test_set_DH = test_set_BS

print("bs_rsmse_list: ", bs_rsmse_list)
print("AVERAGE RSMSE B-S DH: ", np.mean(bs_rsmse_list).item())
print("RSMSE STD B-S DH: ", np.std(bs_rsmse_list).item())

print("leland_rsmse_list: ", leland_rsmse_list)
print("AVERAGE RSMSE LELAND DH: ", np.mean(leland_rsmse_list).item())
print("RSMSE STD LELAND DH: ", np.std(leland_rsmse_list).item())

# For reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initialize Deep Hedging environement
deep_hedging_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                                 nbs_shares, light, train_set=train_set, test_set=test_set[0], trans_costs=trans_costs, discretized=False, device=device)

validation_deep_hedging_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                                     nbs_shares, light, train_set=val_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)

"""Train and test PG"""
def train_test_pg(train=False):
    config={
        "lr": 0.00001,
        "batch_size": 256,
        "num_layers": 4,
        "hidden_size": 64,
    }

    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/pg_hyperparameters/" + time_frame + "/" + str(trans_costs) + "/"

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    deep_hedging_env.discretized = False
    validation_deep_hedging_env.discretized = False
    pg_agent = PG.PG(config=config, state_size=state_size, action_size=1, device=device)
    if train:
        start_time = datetime.datetime.now()
        pg_train_losses = pg_agent.train(deep_hedging_env, validation_deep_hedging_env, episodes=episodes, BS_rsmse=rsmse_DH_leland, lr_schedule=lr_schedule, render=True)
        time_taken = str(datetime.datetime.now() - start_time)
        pg_agent.save(hyperparameter_path + "best_pg_model.pth")
    else:
        pg_agent.load(hyperparameter_path + "best_pg_model.pth")
    pg_rsmse_list = []
    for i in range(num_test_sets):
        deep_hedging_env.test_set = test_set[i].to(device)
        if i == 0:
            pg_actions, pg_rewards, pg_rsmse = pg_agent.test(deep_hedging_env)
        else:
            _, _, pg_rsmse = pg_agent.test(deep_hedging_env)
        pg_rsmse_list.append(pg_rsmse)
    print("pg_rsmse_list: ", pg_rsmse_list)
    print("TEST SET PERFORMANCE: " + str(np.mean(pg_rsmse_list).item()) + " +- " + str(np.std(pg_rsmse_list).item()))
    if train:
        pg_performance = "TEST SET PERFORMANCE: " + str(np.mean(pg_rsmse_list).item()) + " +- " + str(np.std(pg_rsmse_list).item()) + " | TIME TAKEN: " + time_taken
        print("TIME TAKEN: " + time_taken)
        
        with open(hyperparameter_path + "pg_performance.txt", 'w') as file:
            file.write(pg_performance)

        with open(hyperparameter_path + "pg_train_losses_best_model.pickle", 'wb') as file:
            pickle.dump(pg_train_losses, file)

        with open(hyperparameter_path + "pg_train_losses_best_model.pickle", "rb") as file:
            pg_train_losses = pickle.load(file)

        pg_train_losses_fig = plt.figure(figsize=(12, 6))
        plt.plot(pg_train_losses[1:], label="RSMSE")
        plt.xlabel("Episodes (1000s)")
        plt.ylabel("RSMSE")
        plt.legend()
        plt.grid(which="both")
        plt.title("Testing RSMSE for PG")
        plt.savefig(hyperparameter_path + "pg_train_losses_best_model.png")
        plt.close()

    print("POLICY GRADIENT RSMSE: " + str(pg_rsmse))

    pg_actions = pg_actions.cpu().detach().numpy()
    pg_rewards = pg_rewards.cpu().detach().numpy()

    print(" ----------------- ")
    print(" Policy Gradient Results")
    print(" ----------------- ")
    Utils_general.print_stats(pg_rewards, pg_actions, "RSMSE", "Policy Gradient", V_0)

    return pg_actions, pg_rsmse_list

"""Train and test DQN"""
def train_test_dqn(train=False, dueling=False, double=False):
    if dueling and double:
        config={"lr": 0.00001, "batch_size": 128, "num_layers": 4, "hidden_size": 128}
        dqn_model_type = "dueling_double"
    elif dueling:
        config={"lr": 0.00001, "batch_size": 128, "num_layers": 4, "hidden_size": 128}
        dqn_model_type = "dueling"
    elif double:
        config={"lr": 0.0001, "batch_size": 128, "num_layers": 3, "hidden_size": 256}
        dqn_model_type = "double"
    else:
        config={"lr": 0.0001, "batch_size": 256, "num_layers": 3, "hidden_size": 64}
        dqn_model_type = "vanilla"

    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/dqn_hyperparameters/" + dqn_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    deep_hedging_env.discretized = True
    validation_deep_hedging_env.discretized = True
    action_size = deep_hedging_env.discretized_actions.shape[0]

    dqn_agent = DQN.DoubleDQN(config=config, state_size=state_size, action_size=action_size, double=double, dueling=dueling, device=device)
    if train:
        start_time = datetime.datetime.now()
        dqn_train_losses = dqn_agent.train(deep_hedging_env, validation_deep_hedging_env, rsmse_DH_leland, episodes=episodes, lr_schedule=lr_schedule, render=True)
        time_taken = str(datetime.datetime.now() - start_time)
        dqn_agent.save(hyperparameter_path + "best_dqn_model.pth")
    else:
        dqn_agent.load(hyperparameter_path + "best_dqn_model.pth")
    dqn_rsmse_list = []
    for i in range(num_test_sets):
        deep_hedging_env.test_set = test_set[i].to(device)
        if i == 0:
            dqn_actions, dqn_rewards, dqn_rsmse = dqn_agent.test(deep_hedging_env)
        else:
            _, _, dqn_rsmse = dqn_agent.test(deep_hedging_env)
        dqn_rsmse_list.append(dqn_rsmse)
    print("dqn_rsmse_list: ", dqn_rsmse_list)
    print("TEST SET PERFORMANCE: " + str(np.mean(dqn_rsmse_list).item()) + " +- " + str(np.std(dqn_rsmse_list).item()))
    if train:
        dqn_performance = "TEST SET PERFORMANCE: " + str(np.mean(dqn_rsmse_list).item()) + " +- " + str(np.std(dqn_rsmse_list).item()) + " | TIME TAKEN: " + time_taken
        print("TIME TAKEN: " + time_taken)
        
        with open(hyperparameter_path + "dqn_performance.txt", 'w') as file:
            file.write(dqn_performance)
        
        with open(hyperparameter_path + "dqn_train_losses_best_model.pickle", 'wb') as file:
                pickle.dump(dqn_train_losses, file)

        with open(hyperparameter_path + "dqn_train_losses_best_model.pickle", "rb") as file:
                dqn_train_losses = pickle.load(file)

        dqn_train_losses_fig = plt.figure(figsize=(12, 6))
        plt.plot(dqn_train_losses, label="RSMSE")
        plt.xlabel("Episodes (1000s)")
        plt.ylabel("RSMSE")
        plt.legend()
        plt.grid(which="both")
        plt.title("Testing RSMSE for DQN")
        plt.savefig(hyperparameter_path + "dqn_train_losses_best_model.png")
        plt.close()

    print("DQN RSMSE: " + str(dqn_rsmse))

    dqn_actions = dqn_actions.cpu().detach().numpy()
    dqn_rewards = dqn_rewards.cpu().detach().numpy()

    print(" ----------------- ")
    print(" DQN Results")
    print(" ----------------- ")
    Utils_general.print_stats(dqn_rewards, dqn_actions, "RSMSE", "DQN", V_0)
    
    return dqn_actions, dqn_rsmse_list

"""Train and test PPO"""
def train_test_ppo(train=False):
    config={
        "lr": 0.00001,
        "batch_size": 256,
        "num_layers": 2,
        "hidden_size": 256,
    }

    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/ppo_hyperparameters/" + time_frame + "/" + str(trans_costs) + "/"

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    deep_hedging_env.discretized = False
    validation_deep_hedging_env.discretized = False
    ppo_agent = PPO.PPO(config=config, state_size=state_size, action_size=1, device=device)
    if train:
        start_time = datetime.datetime.now()
        ppo_train_losses = ppo_agent.train(deep_hedging_env, validation_deep_hedging_env, rsmse_DH_leland, episodes=episodes, lr_schedule=lr_schedule, render=True)
        time_taken = str(datetime.datetime.now() - start_time)
        ppo_agent.save(hyperparameter_path + "best_ppo_model.pth")
    else:
        ppo_agent.load(hyperparameter_path + "best_ppo_model.pth")
    ppo_rsmse_list = []
    for i in range(num_test_sets):
        deep_hedging_env.test_set = test_set[i].to(device)
        if i == 0:
            ppo_actions, ppo_rewards, ppo_rsmse = ppo_agent.test(deep_hedging_env)
        else:
            _, _, ppo_rsmse = ppo_agent.test(deep_hedging_env)
        ppo_rsmse_list.append(ppo_rsmse)
    print("ppo_rsmse_list: ", ppo_rsmse_list)
    print("TEST SET PERFORMANCE: " + str(np.mean(ppo_rsmse_list).item()) + " +- " + str(np.std(ppo_rsmse_list).item()))
    
    if train:
        ppo_performance = "TEST SET PERFORMANCE: " + str(np.mean(ppo_rsmse_list).item()) + " +- " + str(np.std(ppo_rsmse_list).item()) + " | TIME TAKEN: " + time_taken
        print("TIME TAKEN: " + time_taken)
        
        with open(hyperparameter_path + "ppo_performance.txt", 'w') as file:
            file.write(ppo_performance)
        
        with open(hyperparameter_path + "ppo_train_losses_best_model.pickle", 'wb') as file:
                pickle.dump(ppo_train_losses, file)

        with open(hyperparameter_path + "ppo_train_losses_best_model.pickle", "rb") as file:
                ppo_train_losses = pickle.load(file)

        ppo_train_losses_fig = plt.figure(figsize=(12, 6))
        plt.plot(ppo_train_losses, label="RSMSE")
        plt.xlabel("Episodes")
        plt.ylabel("RSMSE")
        plt.legend()
        plt.grid(which="both")
        plt.title("Validation RSMSE for PPO")
        plt.savefig(hyperparameter_path + "ppo_train_losses_best_model.png")
        plt.close()

    print("PROXIMAL POLICY OPTIMIZATION RSMSE: " + str(ppo_rsmse))

    ppo_actions = ppo_actions.cpu().detach().numpy()
    ppo_rewards = ppo_rewards.cpu().detach().numpy()

    print(" ----------------- ")
    print(" Proximal Policy Optimization Results")
    print(" ----------------- ")
    Utils_general.print_stats(ppo_rewards, ppo_actions, "RSMSE", "Proximal Policy Optimization", V_0)

    return ppo_actions, ppo_rsmse_list

"""Train and test DDPG"""
def train_test_ddpg(train=False, twin_delayed=twin_delayed):
    if twin_delayed:
        config={"lr": 0.00001, "batch_size": 64, "num_layers": 4, "hidden_size": 256}
        ddpg_model_type = "twin_delayed"
    else:
        config={"lr": 0.00001, "batch_size": 64, "num_layers": 4, "hidden_size": 256}
        ddpg_model_type = "vanilla"
    
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/ddpg_hyperparameters/" + ddpg_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    deep_hedging_env.discretized = False
    validation_deep_hedging_env.discretized = False
    ddpg_agent = DDPG.DDPG(config=config, state_size=state_size, action_size=1, twin_delayed=twin_delayed, device=device)
    if train:
        start_time = datetime.datetime.now()
        ddpg_train_losses = ddpg_agent.train(deep_hedging_env, validation_deep_hedging_env, rsmse_DH_leland, episodes=episodes, lr_schedule=lr_schedule, render=True)
        time_taken = str(datetime.datetime.now() - start_time)
        ddpg_agent.save(hyperparameter_path + "best_ddpg_model.pth")
    else:
        ddpg_agent.load(hyperparameter_path + "best_ddpg_model.pth")
    ddpg_rsmse_list = []
    for i in range(num_test_sets):
        deep_hedging_env.test_set = test_set[i].to(device)
        if i == 0:
            ddpg_actions, ddpg_rewards, ddpg_rsmse = ddpg_agent.test(deep_hedging_env)
        else:
            _, _, ddpg_rsmse = ddpg_agent.test(deep_hedging_env)
        ddpg_rsmse_list.append(ddpg_rsmse)
    print("ddpg_rsmse_list: ", ddpg_rsmse_list)
    print("TEST SET PERFORMANCE: " + str(np.mean(ddpg_rsmse_list).item()) + " +- " + str(np.std(ddpg_rsmse_list).item()))

    if train:
        ddpg_performance = "TEST SET PERFORMANCE: " + str(np.mean(ddpg_rsmse_list).item()) + " +- " + str(np.std(ddpg_rsmse_list).item()) + " | TIME TAKEN: " + time_taken
        print("TIME TAKEN: " + time_taken)
        with open(hyperparameter_path + "ddpg_performance.txt", 'w') as file:
            file.write(ddpg_performance)
        
        with open(hyperparameter_path + "ddpg_train_losses_best_model.pickle", 'wb') as file:
                pickle.dump(ddpg_train_losses, file)

        with open(hyperparameter_path + "ddpg_train_losses_best_model.pickle", "rb") as file:
                ddpg_train_losses = pickle.load(file)

        ddpg_train_losses_fig = plt.figure(figsize=(12, 6))
        plt.plot(ddpg_train_losses, label="RSMSE")
        plt.xlabel("Episodes")
        plt.ylabel("RSMSE")
        plt.legend()
        plt.grid(which="both")
        plt.title("Validation RSMSE for DDPG")
        plt.savefig(hyperparameter_path + "ddpg_train_losses_best_model.png")
        plt.close()

    print("DDPG OPTIMIZATION RSMSE: " + str(ddpg_rsmse))

    ddpg_actions = ddpg_actions.cpu().detach().numpy()
    ddpg_rewards = ddpg_rewards.cpu().detach().numpy()

    print(" ----------------- ")
    print(" DDPG Optimization Results")
    print(" ----------------- ")
    Utils_general.print_stats(ddpg_rewards, ddpg_actions, "RSMSE", "DDPG", V_0)

    return ddpg_actions, ddpg_rsmse_list

"""HYPERPARAMETER TUNING USING RAYTUNE"""

env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                    nbs_shares, light, train_set=train_set, test_set=test_set[0], trans_costs=trans_costs, discretized=False, device=device)
val_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                        nbs_shares, light, train_set=val_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)

configs={
    "lr": tune.grid_search([0.001, 0.0001, 0.00001]),
    "batch_size": tune.grid_search([64, 128, 256]),
    "num_layers": tune.grid_search([2, 3, 4]),
    "hidden_size": tune.grid_search([64, 128, 256])
}

# configs={
#     "lr": tune.grid_search([0.001, 0.0001, 0.00001]),
#     "batch_size": tune.grid_search([256, 512, 1024]),
#     "num_layers": tune.grid_search([3, 4]),
#     "hidden_size": tune.grid_search([64, 128, 256])
# }

def train_pg(config):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    subprocess.Popen("gpustat")
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/pg_hyperparameters/" + time_frame + "/" + str(trans_costs) + "/"

    env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                        nbs_shares, light, train_set=train_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)
    val_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                            nbs_shares, light, train_set=val_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)
    
    model = PG.PG(config, state_size, action_size=1, device=device)
    train_losses = model.train(env, val_env, BS_rsmse=rsmse_DH_leland, episodes=episodes, lr_schedule=lr_schedule)
    
    config_str = "lr=" + str(model.lr) + "|batch_size=" + str(model.batch_size) + "|num_layers=" + str(model.num_layers) + "|hidden_size=" + str(model.hidden_size)
    with open(hyperparameter_path + "train_losses/" + "pg_train_losses_" + config_str + ".pickle", 'wb') as file:
        pickle.dump(train_losses, file)
    
    _, _, rsmse = model.test(val_env)

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # This saves the model to the trial directory
        model.save(os.path.join(temp_checkpoint_dir, "pg_model.pth"))
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        # Send the current training result back to Tune
        train.report({"rsmse": rsmse}, checkpoint=checkpoint)

def train_dqn(config):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    subprocess.Popen("gpustat")
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/dqn_hyperparameters/" + dqn_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"
    
    env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                        nbs_shares, light, train_set=train_set, test_set=val_set, trans_costs=trans_costs, discretized=True, device=device)
    val_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                            nbs_shares, light, train_set=val_set, test_set=val_set, trans_costs=trans_costs, discretized=True, device=device)

    action_size = env.discretized_actions.shape[0]

    model = DQN.DoubleDQN(config, state_size, action_size=action_size, double=double, dueling=dueling, device=device)
    train_losses = model.train(env, val_env, BS_rsmse=rsmse_DH_leland, episodes=episodes, lr_schedule=lr_schedule)
    
    config_str = "lr=" + str(model.lr) + "|batch_size=" + str(model.batch_size) + "|num_layers=" + str(model.num_layers) + "|hidden_size=" + str(model.hidden_size)
    with open(hyperparameter_path + "train_losses/" + "dqn_train_losses_" + config_str + ".pickle", 'wb') as file:
        pickle.dump(train_losses, file)
    
    _, _, rsmse = model.test(val_env)
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # This saves the model to the trial directory
        model.save(os.path.join(temp_checkpoint_dir, "dqn_model.pth"))
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        # Send the current training result back to Tune
        train.report({"rsmse": rsmse}, checkpoint=checkpoint)

def train_ppo(config):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    subprocess.Popen("gpustat")
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/ppo_hyperparameters/" + time_frame + "/" + str(trans_costs) + "/"

    env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                        nbs_shares, light, train_set=train_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)
    val_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                            nbs_shares, light, train_set=val_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)

    model = PPO.PPO(config, state_size, action_size=1, device=device)
    train_losses = model.train(env, val_env, BS_rsmse=rsmse_DH_leland, episodes=episodes, lr_schedule=lr_schedule)
    
    config_str = "lr=" + str(model.lr) + "|batch_size=" + str(model.batch_size) + "|num_layers=" + str(model.num_layers) + "|hidden_size=" + str(model.hidden_size)
    with open(hyperparameter_path + "train_losses/" + "ppo_train_losses_" + config_str + ".pickle", 'wb') as file:
        pickle.dump(train_losses, file)
    
    _, _, rsmse = model.test(val_env)

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # This saves the model to the trial directory
        model.save(os.path.join(temp_checkpoint_dir, "ppo_model.pth"))
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        # Send the current training result back to Tune
        train.report({"rsmse": rsmse}, checkpoint=checkpoint)

def train_ddpg(config):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    subprocess.Popen("gpustat")
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/ddpg_hyperparameters/" + ddpg_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"

    env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                        nbs_shares, light, train_set=train_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)
    val_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                            nbs_shares, light, train_set=val_set, test_set=val_set, trans_costs=trans_costs, discretized=False, device=device)

    model = DDPG.DDPG(config, state_size, action_size=1, twin_delayed=twin_delayed, device=device)
    train_losses = model.train(env, val_env, BS_rsmse=rsmse_DH_leland, episodes=episodes, lr_schedule=lr_schedule)
    
    config_str = "lr=" + str(model.lr) + "|batch_size=" + str(model.batch_size) + "|num_layers=" + str(model.num_layers) + "|hidden_size=" + str(model.hidden_size)
    with open(hyperparameter_path + "train_losses/" + "ddpg_train_losses_" + config_str + ".pickle", 'wb') as file:
        pickle.dump(train_losses, file)
    
    _, _, rsmse = model.test(val_env)

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # This saves the model to the trial directory
        model.save(os.path.join(temp_checkpoint_dir, "ddpg_model.pth"))
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        # Send the current training result back to Tune
        train.report({"rsmse": rsmse}, checkpoint=checkpoint)

def raytune(train_func, configs, model_name):
    if cpu:
        ray.init(num_cpus=num_cpus)
        trainable_with_gpu = tune.with_resources(train_func, {"cpu": cpus})
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
        trainable_with_gpu = tune.with_resources(train_func, {"gpu": gpus})

    tuner = tune.Tuner(
        trainable_with_gpu,
        param_space=configs,
        tune_config=tune.TuneConfig(
            metric="rsmse",
            mode="min",
        ),
    )

    results = tuner.fit()

    # Get a dataframe of results for a specific score or mode
    df = results.get_dataframe(filter_metric="rsmse", filter_mode="min").sort_values(by=["rsmse"])
    
    if model_name == "ddpg":
        hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/ddpg_hyperparameters/" + ddpg_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"
    elif model_name == "dqn":
        hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/dqn_hyperparameters/" + dqn_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"
    else: 
        hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/" + model_name + "_hyperparameters/" + time_frame + "/" + str(trans_costs) + "/"
    
    with open(hyperparameter_path + "raytune_" + model_name + ".txt", "w") as raytune_file:
        raytune_file.write(df.to_string())

    best_result = results.get_best_result()  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.path  # Get best trial's result directory
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint

    print("best_result: ", best_result)
    print("best_config: ", best_config)
    print("best_logdir: ", best_logdir)
    print("best_checkpoint: ", best_checkpoint)

    shutil.copyfile(best_checkpoint.path + "/" + model_name + "_model.pth", hyperparameter_path + model_name + "_model.pth")
    
    return best_config

# best_config = raytune(train_func=train_pg, configs=configs, model_name="pg")

"""Helper method to print training losses"""

def plot_training_losses(train_losses, model_name, hyperparameter_path):
    train_losses_fig = plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="RSMSE")
    plt.xlabel("Episodes (1000s)")
    # plt.xscale("log")
    plt.ylabel("RSMSE")
    plt.legend()
    plt.grid(which="both")
    plt.title("Validation RSMSE for " + model_name)
    plt.savefig(hyperparameter_path + model_name + "_train_losses.png")
    plt.close()

"""Hyperparameter tuning and testing/plotting best model"""

def tune_pg():
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/pg_hyperparameters/" + time_frame + "/" + str(trans_costs) + "/"
    best_config = raytune(train_func=train_pg, configs=configs, model_name="pg")
    model = PG.PG(best_config, state_size, action_size=1, device=device)
    model.load(hyperparameter_path + "pg_model.pth")
    _, _, rsmse = model.test(env)

    print("rsmse: ", rsmse)

    config_str = "lr=" + str(best_config["lr"]) + "|batch_size=" + str(best_config["batch_size"]) + "|num_layers=" + str(best_config["num_layers"]) + "|hidden_size=" + str(best_config["hidden_size"])

    with open(hyperparameter_path + "train_losses/" + "pg_train_losses_" + config_str + ".pickle", "rb") as file:
        pg_train_losses = pickle.load(file)

    plot_training_losses(pg_train_losses, model_name="pg", hyperparameter_path=hyperparameter_path)

def tune_dqn():
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/dqn_hyperparameters/" + dqn_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"
    best_config = raytune(train_func=train_dqn, configs=configs, model_name="dqn")
        
    env.discretized = True
    action_size = env.discretized_actions.shape[0]
    model = DQN.DoubleDQN(best_config, state_size, action_size=action_size, double=double, dueling=dueling, device=device)
    model.load(hyperparameter_path + "dqn_model.pth")
    _, _, rsmse = model.test(env)

    print("rsmse: ", rsmse)

    config_str = "lr=" + str(best_config["lr"]) + "|batch_size=" + str(best_config["batch_size"]) + "|num_layers=" + str(best_config["num_layers"]) + "|hidden_size=" + str(best_config["hidden_size"])

    with open(hyperparameter_path + "train_losses/" + "dqn_train_losses_" + config_str + ".pickle", "rb") as file:
        dqn_train_losses = pickle.load(file)

    plot_training_losses(dqn_train_losses, model_name="dqn", hyperparameter_path=hyperparameter_path)
    env.discretized = False

def tune_ppo():
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/ppo_hyperparameters/" + time_frame + "/" + str(trans_costs) + "/"
    best_config = raytune(train_func=train_ppo, configs=configs, model_name="ppo")
    model = PPO.PPO(best_config, state_size, action_size=1, device=device)
    model.load(hyperparameter_path + "ppo_model.pth")
    _, _, rsmse = model.test(env)

    print("rsmse: ", rsmse)

    config_str = "lr=" + str(best_config["lr"]) + "|batch_size=" + str(best_config["batch_size"]) + "|num_layers=" + str(best_config["num_layers"]) + "|hidden_size=" + str(best_config["hidden_size"])

    with open(hyperparameter_path + "train_losses/" + "ppo_train_losses_" + config_str + ".pickle", "rb") as file:
        ppo_train_losses = pickle.load(file)

    plot_training_losses(ppo_train_losses, model_name="ppo", hyperparameter_path=hyperparameter_path)

def tune_ddpg():
    hyperparameter_path = global_path_prefix + "option_hedging/hyperparameters/ddpg_hyperparameters/" + ddpg_model_type + "/" + time_frame + "/" + str(trans_costs) + "/"
    best_config = raytune(train_func=train_ddpg, configs=configs, model_name="ddpg")
    model = DDPG.DDPG(best_config, state_size, action_size=1, device=device)
    model.load(hyperparameter_path + "ddpg_model.pth")
    _, _, rsmse = model.test(env)

    print("rsmse: ", rsmse)

    config_str = "lr=" + str(best_config["lr"]) + "|batch_size=" + str(best_config["batch_size"]) + "|num_layers=" + str(best_config["num_layers"]) + "|hidden_size=" + str(best_config["hidden_size"])

    with open(hyperparameter_path + "train_losses/" + "ddpg_train_losses_" + config_str + ".pickle", "rb") as file:
        ddpg_train_losses = pickle.load(file)

    plot_training_losses(ddpg_train_losses, model_name="ddpg", hyperparameter_path=hyperparameter_path)

"""Plotting actions of models for a single underlying stock price path"""
def plot_actions(path, BS_actions, model_actions, model_labels):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    linestyle_tuple = [
    # ('loosely dotted',        (0, (1, 10))),
    # ('dotted',                (0, (1, 5))),
    # ('longer dash with offset', (5, (30, 3))),
    ('long dash with offset', (5, (10, 3))),
    ('densely dotted',        (0, (1, 1))),

    # ('loosely dashed',        (0, (5, 10))),
    # ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),

    # ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    # ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (5, 1, 1, 1))),

    # ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (5, 1, 1, 1, 1, 1)))
    ]
    ax1.grid("both")
    ax1.set_xlabel("Time Step $(t)$", fontsize=13)
    ax1.set_ylabel("Hedging Position $(X_{t+1})$", fontsize=13)
    ax1.plot(BS_actions, label="B-S DH (Baseline)", linewidth=2.0, linestyle=linestyle_tuple[0][1])
    for i, model_label in enumerate(model_labels):
        ax1.plot(model_actions[i], label=model_label, linewidth=2.0, linestyle=linestyle_tuple[i+1][1])
    plt.legend(loc="upper left", fontsize=11)
    ax2 = ax1.twinx()
    # ax2.set_ylim([95, 155])
    color = "k"
    ax2.set_ylabel("Underlying Stock Price $(S_t)$", fontsize=13, color=color)
    ax2.plot(path, label="Underlying Stock Price", linewidth=2.0, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    # plt.title("Underlying Stock Price and Hedging Strategy")
    plt.savefig(global_path_prefix +  "option_hedging/hedging_ratios.png")
    plt.savefig(global_path_prefix +  "option_hedging/hedging_ratios.svg")
    plt.close()

def plot_training_losses(train_losses, model_name, hyperparameter_path):
    train_losses_fig = plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="RSMSE")
    plt.xlabel("Episodes (1000s)")
    # plt.xscale("log")
    plt.ylabel("RSMSE")
    plt.legend()
    plt.grid(which="both")
    plt.title("Validation RSMSE for " + model_name)
    plt.savefig(hyperparameter_path + model_name + "_train_losses.png")
    plt.close()

def plot_ppo_losses():
    first_path = global_path_prefix + "option_hedging/hyperparameters/ppo_hyperparameters/year/0.0/train_losses/ppo_train_losses_lr=0.0001|batch_size=256|num_layers=2|hidden_size=128.pickle"
    second_path = global_path_prefix + "option_hedging/hyperparameters/ppo_hyperparameters/year/0.0/train_losses/ppo_train_losses_lr=0.0001|batch_size=256|num_layers=4|hidden_size=64.pickle"
    third_path = global_path_prefix + "option_hedging/hyperparameters/ppo_hyperparameters/year/0.0/train_losses/ppo_train_losses_lr=1e-05|batch_size=256|num_layers=4|hidden_size=256.pickle"
    fourth_path = global_path_prefix + "option_hedging/hyperparameters/ppo_hyperparameters/year/0.0/train_losses/ppo_train_losses_lr=1e-05|batch_size=256|num_layers=2|hidden_size=256.pickle"
    with open(first_path, "rb") as file:
        first = pickle.load(file)
    with open(second_path, "rb") as file:
        second = pickle.load(file)
    with open(third_path, "rb") as file:
        third = pickle.load(file)
    with open(fourth_path, "rb") as file:
        fourth = pickle.load(file)
    train_losses_fig = plt.figure(figsize=(12, 6))
    # plt.plot(first, label="first hyperparameter combination")
    # plt.plot(second, label="second hyperparameter combination")
    plt.plot(third, label="third hyperparameter combination")
    # plt.plot(fourth, label="fourth hyperparameter combination")
    plt.xlabel("Updates (1000s)")
    # plt.yscale("log")
    plt.ylabel("log RSQP")
    plt.legend()
    plt.grid(which="both")
    plt.title("Validation RSQP for PPO over time")
    plt.savefig(global_path_prefix + "option_hedging/ppo_validation_losses.png")
    plt.close()

"""Get actions from all models"""
discretized_actions = np.arange(start=0.0, stop=1.0, step=0.02)

# pg_actions, pg_test_losses = train_test_pg(train=True)

# dqn_actions_indices, dqn_test_losses = train_test_dqn(train=True, dueling=False, double=False)
# dqn_actions = discretized_actions[dqn_actions_indices.astype(int)]

# double_dqn_actions_indices, double_dqn_test_losses = train_test_dqn(train=True, dueling=False, double=True)
# double_dqn_actions = discretized_actions[double_dqn_actions_indices.astype(int)]

# dueling_dqn_actions_indices, dueling_dqn_test_losses = train_test_dqn(train=True, dueling=True, double=False)
# dueling_dqn_actions = discretized_actions[dueling_dqn_actions_indices.astype(int)]

# dueling_double_dqn_actions_indices, dueling_double_dqn_test_losses = train_test_dqn(train=True, dueling=True, double=True)
# dueling_double_dqn_actions = discretized_actions[dueling_double_dqn_actions_indices.astype(int)]

# ppo_actions, ppo_test_losses = train_test_ppo(train=True)

# ddpg_actions, ddpg_test_losses = train_test_ddpg(train=True)

# td_ddpg_actions, td_ddpg_test_losses = train_test_ddpg(train=True, twin_delayed=True)



# all_algorithms_test_losses = [pg_test_losses, bs_rsmse_list, ppo_test_losses, dueling_dqn_test_losses, td_ddpg_test_losses, ddpg_test_losses, dueling_double_dqn_test_losses, dqn_test_losses, double_dqn_test_losses]

# with open(global_path_prefix + "option_hedging/all_test_losses.pickle", 'wb') as file:
#     pickle.dump(all_algorithms_test_losses, file)
# with open(global_path_prefix + "option_hedging/all_test_losses.pickle", "rb") as file:
#     all_algorithms_test_losses = pickle.load(file)

# all_algorithms_names = ["PG", "B-S DH", "PPO", "Dueling DQL",  "TD3", "DDPG", "Dueling Double DQL", "DQL", "Double DQL"]

# top_row = "||"
# for name in all_algorithms_names:
#     top_row += "{:20s}|".format(name)
# print(top_row)
# for i, test_losses_i in enumerate(all_algorithms_test_losses):
#     current_row = "{:20s} & ".format(all_algorithms_names[i])
#     for j, test_losses_j in enumerate(all_algorithms_test_losses):
#         p_value = ttest_ind(test_losses_i, test_losses_j, equal_var=False, alternative="less").pvalue
#         current_row += "{:.2f} & ".format(p_value)
#     print(current_row)

# model_actions = np.stack([pg_actions, dqn_actions, double_dqn_actions, dueling_dqn_actions, dueling_double_dqn_actions, ppo_actions, ddpg_actions, td_ddpg_actions])

# with open(global_path_prefix + "option_hedging/model_actions.pickle", 'wb') as file:
#     pickle.dump(model_actions, file)

# with open(global_path_prefix + "option_hedging/model_actions.pickle", "rb") as file:
#     model_actions = pickle.load(file)

# model_actions = np.stack([model_actions[0,:,:], model_actions[3,:,:], model_actions[5,:,:], model_actions[7,:,:]])
# model_labels = ["PG", "Dueling DQL", "PPO", "TD3"]

# plot_actions(test_set_DH[:,0], bs_actions[:,0], model_actions[:,:,0], model_labels)
# plot_ppo_losses()







# tune_dqn()
# tune_ddpg()
tune_ppo()







# """Print baseline models performance statistics"""

# print(" ----------------- ")
# print(" Delta Hedging Results")
# print(" ----------------- ")
# deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, 
#                                                           nbs_shares=nbs_shares, trans_costs=trans_costs)
# Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)
# semi_square_hedging_err_DH = np.square(np.where(hedging_err_DH > 0, hedging_err_DH, 0))
# smse_DH = np.mean(semi_square_hedging_err_DH)
# rsmse_DH = np.sqrt(np.mean(semi_square_hedging_err_DH))

# print(" ----------------- ")
# print("Leland Delta Hedging Results")
# print(" ----------------- ")
# deltas_DH_leland, hedging_err_DH_leland = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0,
#                                                                         nbs_shares=nbs_shares, trans_costs=trans_costs, Leland=True)
# Utils_general.print_stats(hedging_err_DH_leland, deltas_DH_leland, "Leland delta hedge", "Leland delta hedge", V_0)
# semi_square_hedging_err_DH_leland = np.square(np.where(hedging_err_DH_leland > 0, hedging_err_DH_leland, 0))
# smse_DH_leland = np.mean(semi_square_hedging_err_DH_leland)
# rsmse_DH_leland = np.sqrt(np.mean(semi_square_hedging_err_DH_leland))

# total_time_taken = str(datetime.datetime.now() - start_total_time)
# print("TOTAL TIME TAKEN: ", total_time_taken)
