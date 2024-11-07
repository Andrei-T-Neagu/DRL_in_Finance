import datetime as datetime
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import matplotlib.pyplot as plt
import Utils_general
import DeepHedgingEnvironment
import RL_algorithms.DQN as DQN
import RL_algorithms.PG as PG
import RL_algorithms.PPO as PPO
import RL_algorithms.DDPG as DDPG
from data_generation_processes.GARCH import GARCH
from scipy.stats import ttest_ind
from scipy.stats import f
import yfinance as yf
global_path_prefix = "/home/a_eagu/DRL_in_Finance/option_hedging/code_pytorch/"

nbs_point_traj = 13
T = 252/252

train_size = 2**20
test_size = 2**17

r_borrow = 0
r_lend = 0

S_0 = 100
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 100

batch_size = 128
num_layers = 3
nbs_units = 128
lr = 0.0001

prepro_stock = "log-moneyness"
nbs_shares = 1

#real market data parameters for garch model
stock = "^GSPC"

start="2000-11-15"
end="2024-10-15"
interval= "1mo"              # Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
garch_type="gjr"

# neural network type parameters
light = True
lr_schedule = True

state_size = 3 if light else 4
trans_costs = 0.00

# Black-Scholes mu and sigma parameters estimated from real market data
market_data = yf.download(stock, start=start, end=end, interval="1d")
log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
mu = log_returns.mean() * 252
sigma = log_returns.std() * np.sqrt(252)
params_vect = [mu, sigma]

# For reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initializing the inital option price using black-scholes option pricing
if (option_type == 'call'):
    # V_0 = 0.0
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    # V_0 = 0.0
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

# Initialize the garch model
garch_model = GARCH(stock=stock, start=start, end=end, interval=interval, type=garch_type)

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
    print("Generating GARCH Data Set")
    dataset = garch_model.generate(S_0=S_0, batch_size=size, num_points=nbs_point_traj, load_params=True)
    dataset = torch.from_numpy(dataset).to(torch.float)
    torch.save(dataset, global_path_prefix + str(dataset_type))

"""Training the garch model and generating the datasets"""
# train_garch()
# generate_garch_dataset(dataset_type="train_set", size=train_size)
# generate_garch_dataset(dataset_type="test_set", size=test_size)

# Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the training and testing datasets
train_set = torch.load(global_path_prefix + "train_set", weights_only=True)
test_set = torch.load(global_path_prefix + "test_set", weights_only=True)

# For reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

"""Print baseline models performance statistics"""

test_set_BS = test_set.detach().cpu().numpy().T

print(" ----------------- ")
print(" Delta Hedging Results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, 
                                                          nbs_shares=nbs_shares, trans_costs=trans_costs)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)
semi_square_hedging_err_DH = np.square(np.where(hedging_err_DH > 0, hedging_err_DH, 0))
smse_DH = np.mean(semi_square_hedging_err_DH)
rsmse_DH = np.sqrt(np.mean(semi_square_hedging_err_DH))


print(" ----------------- ")
print("Leland Delta Hedging Results")
print(" ----------------- ")
deltas_DH_leland, hedging_err_DH_leland = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0,
                                                                        nbs_shares=nbs_shares, trans_costs=trans_costs, Leland=True)
Utils_general.print_stats(hedging_err_DH_leland, deltas_DH_leland, "Leland delta hedge", "Leland delta hedge", V_0)
semi_square_hedging_err_DH_leland = np.square(np.where(hedging_err_DH_leland > 0, hedging_err_DH_leland, 0))
smse_DH_leland = np.mean(semi_square_hedging_err_DH_leland)
rsmse_DH_leland = np.sqrt(np.mean(semi_square_hedging_err_DH_leland))

# For reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initialize Deep Hedging environement
deep_hedging_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                                 nbs_shares, light, train_set=train_set, test_set=test_set, discretized=False)

validation_deep_hedging_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                                     nbs_shares, light, train_set=train_set, test_set=test_set, discretized=False)

lr_list = [0.001, 0.0001, 0.00001]
num_layers_list = [4, 3, 2]
nbs_units_list = [256, 128, 64]
batch_size_list = [256, 128, 64]

episodes = 20000
ma_size = 1000

"""Train and test PG"""

# deep_hedging_env.discretized = False
# pg_agent = PG.PG(state_size=state_size, action_size=1, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
# pg_train_losses = pg_agent.train(deep_hedging_env, validation_deep_hedging_env, episodes=episodes, BS_rsmse=rsmse_DH_leland)
# pg_actions, pg_rewards, pg_rsmse = pg_agent.test(deep_hedging_env)

# hyperparameter_path = "/home/a_eagu/DRL_in_Finance/pg_hyperparameters/"
# pg_losses = np.convolve(pg_train_losses[1000:], np.ones(ma_size), 'valid') / ma_size
# pg_train_losses_fig = plt.figure(figsize=(12, 6))
# plt.plot(pg_losses, label="RSMSE")
# plt.xlabel("Episodes")
# # plt.xscale("log")
# plt.ylabel("RSMSE")
# plt.legend()
# plt.grid(which="both")
# plt.title("RSMSE " + str(ma_size) + " Episode Moving Average for PG")
# plt.savefig(hyperparameter_path + "training_losses/pg_train_losses.png")
# plt.close()

# print("POLICY GRADIENT RSMSE: " + str(pg_rsmse))

# pg_actions = pg_actions.cpu().detach().numpy()
# pg_rewards = pg_rewards.cpu().detach().numpy()

# print(" ----------------- ")
# print(" Policy Gradient Results")
# print(" ----------------- ")
# Utils_general.print_stats(pg_rewards, pg_actions, "RSMSE", "Policy Gradient", V_0)

"""Train and test DQN"""

deep_hedging_env.discretized = True
validation_deep_hedging_env.discretized = True
action_size = deep_hedging_env.discretized_actions.shape[0]
dqn_agent = DQN.DoubleDQN(state_size=state_size, action_size=action_size, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
dqn_train_losses = dqn_agent.train(deep_hedging_env, validation_deep_hedging_env, episodes=episodes, lr_schedule=True)
dqn_actions, dqn_rewards, dqn_rsmse = dqn_agent.test(deep_hedging_env)

hyperparameter_path = "/home/a_eagu/DRL_in_Finance/dqn_hyperparameters/"
dqn_losses = np.convolve(dqn_train_losses, np.ones(ma_size), 'valid') / ma_size
dqn_train_losses_fig = plt.figure(figsize=(12, 6))
plt.plot(dqn_losses, label="RSMSE")
plt.xlabel("Episodes")
plt.ylabel("RSMSE")
plt.legend()
plt.title("RSMSE " + str(ma_size) + " Episode Moving Average for DQN")
plt.savefig(hyperparameter_path + "training_losses/dqn_train_losses.png")
plt.close()

print("DQN RSMSE: " + str(dqn_rsmse))

dqn_actions = dqn_actions.cpu().detach().numpy()
dqn_rewards = dqn_rewards.cpu().detach().numpy()

print(" ----------------- ")
print(" DQN Results")
print(" ----------------- ")
Utils_general.print_stats(dqn_rewards, dqn_actions, "RSMSE", "DQN", V_0)

"""Train and test PPO"""

# deep_hedging_env.discretized = False
# validation_deep_hedging_env.discretized = False
# ppo_agent = PPO.PPO(state_size=state_size, action_size=1, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
# ppo_train_losses = ppo_agent.train(deep_hedging_env, validation_deep_hedging_env, episodes=episodes, lr_schedule=True)
# ppo_actions, ppo_rewards, ppo_rsmse = ppo_agent.test(deep_hedging_env)

# hyperparameter_path = "/home/a_eagu/DRL_in_Finance/ppo_hyperparameters/"
# ppo_losses = np.convolve(ppo_train_losses, np.ones(ma_size), 'valid') / ma_size
# ppo_train_losses_fig = plt.figure(figsize=(12, 6))
# plt.plot(ppo_losses, label="RSMSE")
# plt.xlabel("Episodes")
# plt.ylabel("RSMSE")
# plt.legend()
# plt.title("RSMSE " + str(ma_size) + " Episode Moving Average for PPO")
# plt.savefig(hyperparameter_path + "training_losses/ppo_train_losses.png")
# plt.close()

# print("PROXIMAL POLICY OPTIMIZATION RSMSE: " + str(ppo_rsmse))

# ppo_actions = ppo_actions.cpu().detach().numpy()
# ppo_rewards = ppo_rewards.cpu().detach().numpy()

# print(" ----------------- ")
# print(" Proximal Policy Optimization Results")
# print(" ----------------- ")
# Utils_general.print_stats(ppo_rewards, ppo_actions, "RSMSE", "Proximal Policy Optimization", V_0)

"""Train and test DDPG"""

# deep_hedging_env.discretized = False
# validation_deep_hedging_env.discretized = False
# ddpg_agent = DDPG.DDPG(state_size=state_size, action_size=1, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size, twin_delayed=True)
# ddpg_train_losses = ddpg_agent.train(deep_hedging_env, validation_deep_hedging_env, episodes=episodes, lr_schedule=True)
# ddpg_actions, ddpg_rewards, ddpg_rsmse = ddpg_agent.test(deep_hedging_env)

# hyperparameter_path = "/home/a_eagu/DRL_in_Finance/ddpg_hyperparameters/"
# ddpg_losses = np.convolve(ddpg_train_losses, np.ones(ma_size), 'valid') / ma_size
# ddpg_train_losses_fig = plt.figure(figsize=(12, 6))
# plt.plot(ddpg_losses, label="RSMSE")
# plt.xlabel("Episodes")
# plt.ylabel("RSMSE")
# plt.legend()
# plt.title("RSMSE " + str(ma_size) + " Episode Moving Average for DDPG")
# plt.savefig(hyperparameter_path + "training_losses/ddpg_train_losses.png")
# plt.close()

# print("DDPG OPTIMIZATION RSMSE: " + str(ddpg_rsmse))

# ddpg_actions = ddpg_actions.cpu().detach().numpy()
# ddpg_rewards = ddpg_rewards.cpu().detach().numpy()

# print(" ----------------- ")
# print(" DDPG Optimization Results")
# print(" ----------------- ")
# Utils_general.print_stats(ddpg_rewards, ddpg_actions, "RSMSE", "DDPG", V_0)

"""HYPERPARAMETER TUNING USING RAYTUNE"""

# env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
#                                                                                      nbs_shares, light, train_set=train_set, test_set=test_set, discretized=False)
# val_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
#                                                                                              nbs_shares, light, train_set=train_set, test_set=test_set, discretized=False)

# def train_ddpg(config):
#     torch.manual_seed(0)
#     random.seed(0)
#     np.random.seed(0)
    
#     action_size = 1
    
#     model = DDPG.DDPG(config, state_size, action_size)
#     model.train(env, val_env, episodes=100000)
#     model.test(val_env)

# config={
#     "lr": tune.grid_search([0.0001, 0.00001, 0.000001]),
#     "batch_size": tune.grid_search([64, 128, 256]),
#     "num_layers": tune.grid_search([2, 3, 4]),
#     "hidden_size": tune.grid_search([64, 128, 256])
# }

# ray.init(_temp_dir="/home/a_eagu/DRL_in_Finance/temp")
# trainable_with_gpu = tune.with_resources(train_ddpg, {"gpu": 0.02})

# tuner = tune.Tuner(
#     trainable_with_gpu,
#     param_space=config,
#     tune_config=tune.TuneConfig(
#         scheduler=ASHAScheduler(metric="rsmse", mode="min"),
#     ),
# )

# results = tuner.fit()

# # Get a dataframe of results for a specific score or mode
# df = results.get_dataframe(filter_metric="rsmse", filter_mode="min").sort_values(by=["rsmse"])
# with open("/home/a_eagu/DRL_in_Finance/raytune_ddpg.txt", "w") as raytune_file:
#     raytune_file.write(df.to_string())


"""Print baseline models performance statistics"""

test_set_BS = test_set.detach().cpu().numpy().T

print(" ----------------- ")
print(" Delta Hedging Results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0, 
                                                          nbs_shares=nbs_shares, trans_costs=trans_costs)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)
semi_square_hedging_err_DH = np.square(np.where(hedging_err_DH > 0, hedging_err_DH, 0))
smse_DH = np.mean(semi_square_hedging_err_DH)
rsmse_DH = np.sqrt(np.mean(semi_square_hedging_err_DH))


print(" ----------------- ")
print("Leland Delta Hedging Results")
print(" ----------------- ")
deltas_DH_leland, hedging_err_DH_leland = Utils_general.delta_hedge_res(test_set_BS, r_borrow, r_lend, params_vect[1], T, option_type=option_type, position_type=position_type, strike=strike, V_0=V_0,
                                                                        nbs_shares=nbs_shares, trans_costs=trans_costs, Leland=True)
Utils_general.print_stats(hedging_err_DH_leland, deltas_DH_leland, "Leland delta hedge", "Leland delta hedge", V_0)
semi_square_hedging_err_DH_leland = np.square(np.where(hedging_err_DH_leland > 0, hedging_err_DH_leland, 0))
smse_DH_leland = np.mean(semi_square_hedging_err_DH_leland)
rsmse_DH_leland = np.sqrt(np.mean(semi_square_hedging_err_DH_leland))













"""Grid search hyperparameter tuning"""

def hyperparameter_tuning(agent_type, episodes, lr_list, num_layers_list, nbs_units_list, batch_size_list, discretized=False):
    hyperparameter_path = "/home/a_eagu/DRL_in_Finance/" + agent_type + "_hyperparameters/"
    configs = []
    rsmses = []
    config_index = 0
    total_configs = len(lr_list) * len(num_layers_list) * len(nbs_units_list) * len(batch_size_list) 
    
    total_time_start = datetime.datetime.now()
    
    for lr in lr_list:
        for num_layers in num_layers_list:
            for nbs_units in nbs_units_list:
                for batch_size in batch_size_list:

                    training_start = datetime.datetime.now()

                    config_index += 1
                    config_string = "config: " + str(config_index) + " | lr: " + str(lr) + " | num_layers: " + str(num_layers) + " | nbs_units: " + str(nbs_units) + " | batch_size: " + str(batch_size)
                    configs.append(config_string)
                    
                    print("CONFIGURATION: " + str(config_index) + "\\" + str(total_configs))

                    # For reproducibility
                    torch.manual_seed(0)
                    random.seed(0)
                    np.random.seed(0)

                    # Initialize Deep Hedging environment
                    deep_hedging_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                                                     nbs_shares, light, train_set=train_set, test_set=test_set, discretized=False)

                    if agent_type != "pg":
                        validation_deep_hedging_env = DeepHedgingEnvironment.DeepHedgingEnvironment(nbs_point_traj, r_borrow, r_lend, S_0, T, option_type, position_type, strike, V_0, prepro_stock,
                                                                                             nbs_shares, light, train_set=train_set, test_set=test_set, discretized=False)
                    if discretized:
                        action_size = deep_hedging_env.discretized_actions.shape[0]
                    else:
                        action_size = 1
                    
                    if agent_type == "pg":
                        agent = PG.PG(state_size=state_size, action_size=action_size, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
                    if agent_type == "dqn":
                        agent = DQN.DoubleDQN(state_size=state_size, action_size=action_size, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
                    if agent_type == "ppo":
                        agent = PPO.PPO(state_size=state_size, action_size=action_size, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
                    if agent_type == "ddpg":
                        agent = DDPG.DDPG(state_size=state_size, action_size=action_size, num_layers=num_layers, hidden_size=nbs_units, lr=lr, batch_size=batch_size)
                    
                    if agent_type == "pg":
                        train_losses = agent.train(deep_hedging_env, validation_deep_hedging_env, episodes=episodes, lr_schedule=lr_schedule)
                    else:
                        train_losses = agent.train(deep_hedging_env, validation_deep_hedging_env, episodes=episodes, lr_schedule=lr_schedule)
                    actions, rewards, rsmse = agent.test(deep_hedging_env)
                    
                    rsmses.append(rsmse)
                    
                    actions = actions.cpu().detach().numpy()
                    rewards = rewards.cpu().detach().numpy()

                    ma_losses = np.convolve(train_losses, np.ones(ma_size), 'valid') / ma_size
                    train_losses_fig = plt.figure(figsize=(12, 6))
                    plt.plot(ma_losses, label="RSMSE")
                    plt.xlabel("Episodes")
                    plt.ylabel("RSMSE")
                    plt.legend()
                    plt.title("RSMSE " + str(ma_size) + " Episode Moving Average for " + agent_type + " with " + config_string)
                    plt.savefig(hyperparameter_path + "training_losses/" + agent_type + "_train_losses_CONFIG_"+ str(config_index) + ".png")
                    plt.close()

                    training_end = datetime.datetime.now()
                    print("TIME TAKEN: " + str(training_end-training_start))
                    print("TOTAL TIME TAKEN: " + str(training_end-total_time_start))

    sorted_valid_indices = np.argsort(rsmses).tolist()

    with open(hyperparameter_path + agent_type + "_hyperparameters_file.txt", "w") as hyperparameter_tune_file:
        # Writing data to a file
        for i in sorted_valid_indices:
            hyperparameter_tune_file.write(configs[i] + " | rsmse: " + str(rsmses[i]) + "\n")