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
from scipy.stats import ttest_ind

def count_parameters(agent):
    return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)

nbs_point_traj = 13
time_period = "year"
batch_size = 512
train_size = 100000
test_size = 1000000
valid_size = 1000000
epochs = 100
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.1, 0.1898]
S_0 = 1000
T = 252/252
alpha = 1.02
beta = 0.98
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 2
nbs_units = 256
lr = 0.0001
dropout = 0
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [-1, -1]

light = False
generate_valid_dataset = True
generate_test_dataset = True
lr_schedule = True

if light:
    path_prefix = "code_pytorch/hyperparameter_tuning/light/"

    name_ffnn = 'code_pytorch/ffnn_model_light_' + str(time_period)
else:
    path_prefix = "code_pytorch/hyperparameter_tuning/"

    name_ffnn = 'code_pytorch/ffnn_model_' + str(time_period)

if (option_type == 'call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

if generate_test_dataset:
    # Creating test dataset
    mu, sigma = params_vect
    N = nbs_point_traj - 1
    dt = T / N

    test_set = torch.ones(test_size, nbs_point_traj)
    print("Generating Unbatched Test Data Set")
    for episode in range(test_size):
        if episode % 100000 == 0:
            print("Generated: " + str(episode) + "/" + str(test_size) + " paths")
        for timestep in range(nbs_point_traj):
            if timestep == 0:
                S_t = S_0 
            else:
                Z = torch.randn(1)
                S_t = S_t * torch.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * Z)
            test_set[episode, timestep] = S_t
    
    test_set = torch.permute(test_set, (1, 0))
    torch.save(test_set, "/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/unbatched_test_set")

if generate_valid_dataset:
    valid_set = torch.ones(valid_size, nbs_point_traj)
    print("Generating Unbatched Validation Data Set")
    for episode in range(valid_size):
        if episode % 100000 == 0:
            print("Generated: " + str(episode) + "/" + str(valid_size) + " paths")
        for timestep in range(nbs_point_traj):
            if timestep == 0:
                S_t = S_0 
            else:
                Z = torch.randn(1)
                S_t = S_t * torch.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * Z)
            valid_set[episode, timestep] = S_t

    valid_set = torch.permute(valid_set, (1, 0))
    torch.save(valid_set, "/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/unbatched_valid_set")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_set = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/unbatched_test_set")
test_set = test_set.to(device)

valid_set = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/code_pytorch/unbatched_valid_set")
valid_set = valid_set.to(device)

hyperparameters = [
                   {"batch_size":512, "nbs_layers":8, "nbs_units":1024},
                   {"batch_size":512, "nbs_layers":8, "nbs_units":512},
                   {"batch_size":512, "nbs_layers":8, "nbs_units":256},
                   {"batch_size":512, "nbs_layers":6, "nbs_units":1024},
                   {"batch_size":512, "nbs_layers":6, "nbs_units":512},
                   {"batch_size":512, "nbs_layers":6, "nbs_units":256},
                   {"batch_size":512, "nbs_layers":4, "nbs_units":1024},
                   {"batch_size":512, "nbs_layers":4, "nbs_units":512},
                   {"batch_size":512, "nbs_layers":4, "nbs_units":256},

                   {"batch_size":256, "nbs_layers":8, "nbs_units":1024},
                   {"batch_size":256, "nbs_layers":8, "nbs_units":512},
                   {"batch_size":256, "nbs_layers":8, "nbs_units":256},
                   {"batch_size":256, "nbs_layers":6, "nbs_units":1024},
                   {"batch_size":256, "nbs_layers":6, "nbs_units":512},
                   {"batch_size":256, "nbs_layers":6, "nbs_units":256},
                   {"batch_size":256, "nbs_layers":4, "nbs_units":1024},
                   {"batch_size":256, "nbs_layers":4, "nbs_units":512},
                   {"batch_size":256, "nbs_layers":4, "nbs_units":256},
                   
                   {"batch_size":128, "nbs_layers":8, "nbs_units":1024},
                   {"batch_size":128, "nbs_layers":8, "nbs_units":512},
                   {"batch_size":128, "nbs_layers":8, "nbs_units":256},
                   {"batch_size":128, "nbs_layers":6, "nbs_units":1024},
                   {"batch_size":128, "nbs_layers":6, "nbs_units":512},
                   {"batch_size":128, "nbs_layers":6, "nbs_units":256},
                   {"batch_size":128, "nbs_layers":4, "nbs_units":1024},
                   {"batch_size":128, "nbs_layers":4, "nbs_units":512},
                   {"batch_size":128, "nbs_layers":4, "nbs_units":256},

                   {"batch_size":64, "nbs_layers":8, "nbs_units":1024},
                   {"batch_size":64, "nbs_layers":8, "nbs_units":512},
                   {"batch_size":64, "nbs_layers":8, "nbs_units":256},
                   {"batch_size":64, "nbs_layers":6, "nbs_units":1024},
                   {"batch_size":64, "nbs_layers":6, "nbs_units":512},
                   {"batch_size":64, "nbs_layers":6, "nbs_units":256},
                   {"batch_size":64, "nbs_layers":4, "nbs_units":1024},
                   {"batch_size":64, "nbs_layers":4, "nbs_units":512},
                   {"batch_size":64, "nbs_layers":4, "nbs_units":256},
                  ]

config_losses_per_epoch = []
config_test_losses = []
config_valid_losses = []
training_times = []
num_parameters = []

for i, config in enumerate(hyperparameters):

    test_index = test_size%config["batch_size"]
    if test_index != 0:
        test_set_batched = test_set[:, :-test_index].view(nbs_point_traj, int(test_size/config["batch_size"]),  config["batch_size"])
    else:
        test_set_batched = test_set.view(nbs_point_traj, int(test_size/config["batch_size"]),  config["batch_size"])
    test_set_batched = torch.permute(test_set_batched, (1,0,2))

    valid_index = valid_size%config["batch_size"]
    if test_index != 0:
        valid_set_batched = valid_set[:, :-valid_index].view(nbs_point_traj, int(valid_size/config["batch_size"]),  config["batch_size"])
    else:
        valid_set_batched = valid_set.view(nbs_point_traj, int(valid_size/config["batch_size"]),  config["batch_size"])
    valid_set_batched = torch.permute(valid_set_batched, (1,0,2))
    
    training_start = datetime.datetime.now()
    
    if light:
        agent_ffnn = DeepAgentLight.DeepAgent(nbs_point_traj, config["batch_size"], r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                    loss_type, option_type, position_type, strike, V_0, config["nbs_layers"], config["nbs_units"], lr, dropout, prepro_stock,
                    nbs_shares, lambdas, name=name_ffnn)
    else:
        agent_ffnn = DeepHedgingEnvironment.DeepAgent(nbs_point_traj, config["batch_size"], r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                    loss_type, option_type, position_type, strike, V_0, config["nbs_layers"], config["nbs_units"], lr, dropout, prepro_stock,
                    nbs_shares, lambdas, name=name_ffnn)

    num_parameters.append(count_parameters(agent_ffnn))
    
    print("CURRENTLY TRAINING: " + str(config))
    
    all_losses_ffnn, ffnn_losses = agent_ffnn.train(train_size = train_size, epochs=epochs, lr_schedule = lr_schedule)
    training_times.append(datetime.datetime.now()-training_start)
    agent_ffnn.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_ffnn)

    deltas_ffnn_valid, hedging_err_ffnn_valid, S_t_ffnn_valid, V_t_ffnn_valid, A_t_ffnn_valid, B_t_ffnn_valid = agent_ffnn.test(test_size=valid_size, test_set=valid_set_batched)

    deltas_ffnn_test, hedging_err_ffnn_test, S_t_ffnn_test, V_t_ffnn_test, A_t_ffnn_test, B_t_ffnn_test = agent_ffnn.test(test_size=test_size, test_set=test_set_batched)
    if loss_type == "RSMSE":
        semi_square_hedging_err_ffnn_valid = np.square(np.where(hedging_err_ffnn_valid > 0, hedging_err_ffnn_valid, 0))
        semi_square_hedging_err_ffnn_test = np.square(np.where(hedging_err_ffnn_test > 0, hedging_err_ffnn_test, 0))
        rsmse_ffnn_valid = np.sqrt(np.mean(semi_square_hedging_err_ffnn_valid))
        rsmse_ffnn_test = np.sqrt(np.mean(semi_square_hedging_err_ffnn_test))
        error_valid = rsmse_ffnn_valid
        error_test = rsmse_ffnn_test
    else: 
        rmse_ffnn_valid = np.sqrt(np.mean(hedging_err_ffnn_valid))
        rmse_ffnn_test = np.sqrt(np.mean(hedging_err_ffnn_test))
        error_valid = rmse_ffnn_valid
        error_test = rmse_ffnn_test
    
    config_valid_losses.append(error_valid)
    config_test_losses.append(error_test)
    config_losses_per_epoch.append(ffnn_losses)

sorted_valid_indices = np.argsort(config_valid_losses).tolist()

with open(path_prefix + "ffnn_hyperparameter_tuning.txt", "w") as hyperparameter_tune_file:
    # Writing data to a file
    hyperparameter_tune_file.write("|---------------------------Hyperparameter Tuning for FFNN for {epochs:d} Epochs------------------------|\n".format(epochs=epochs))
    for i in sorted_valid_indices:
        hyperparameter_tune_file.write(str(hyperparameters[i]) + ": Validation Loss: " + str(config_valid_losses[i].item()) + " | Test Loss: " + str(config_test_losses[i].item()) + " | Number of Parameters: " + str(num_parameters[i]) + " | Time Taken: " + str(training_times[i]) + "\n")

epoch_losses_fig = plt.figure(figsize=(20, 10))
for i, config in enumerate(hyperparameters):
    plt.plot(config_losses_per_epoch[i], label=str(config))
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error per Epoch of Hyperparameter Search for FFNN Architecture")
plt.savefig(path_prefix + "ffnn_hyperparameter_search_" + str(time_period) + ".png")