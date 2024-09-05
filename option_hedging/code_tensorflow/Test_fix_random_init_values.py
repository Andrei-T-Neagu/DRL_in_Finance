#################################################
# Resultas are in Results_comp_random_vs_fix_init_values.txt
#################################################



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime as dt
import numpy as np
#import tensorflow as tf             # TENSORFLOW ---> deep learning (installation CPU)
import tensorflow.compat.v1 as tf            # to keep using Alex's code from tensorflow v1
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import random
#from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.optimizers import Adam   # to keep using Alex's code from tensorflow v1

#from keras import backend as K
from tensorflow.compat.v1.keras import backend as K       # to keep using Alex's code from tensorflow v1

from scipy import stats
from scipy.special import ndtri
#import import_ipynb
import importlib

import Utils_general
import DeepAgent_class_fix_init
import DeepAgent_class_rand_init
import utils_cs

tf.compat.v1.disable_eager_execution() #Error with the use of the loss function If not disabled

importlib.reload(Utils_general)
importlib.reload(DeepAgent_class_fix_init)
importlib.reload(DeepAgent_class_rand_init)

##################
# Inputs
##################

# General got every case
T = 1 / 252  # 252 trading days per year
batch_size = 10000  # {128;256;500;1000}
r_borrow = 0.00
r_lend = 0.00
S_0 = 100.0
option_type = 'Call'
position_type = 'Short'  # {'Short', 'Long'}
freq_dyn = 'hourly'
freq_rebal = 'hourly'
if (freq_rebal == "weekly"):
    nbs_point_traj = int(T * 52) + 1
elif (freq_rebal == "daily"):
    nbs_point_traj = int(T * 252) + 1
elif (freq_rebal == "monthly"):
    nbs_point_traj = int(T * 12) + 1
elif freq_rebal == "hourly":
    nbs_point_traj = int(T * 252 * 8) + 1 # We suppose stock markets are open 8 hours a day


# model parameters
stock_dyn = 'BSM'
[mu, sigma] = [0.0, 0.1898]
params_vect = [mu, sigma]
hab = [1397.38, 1397.38] #[9284.023, 9284.023] # for 99% of resilience #[4642.01, 4642.01]   for 50%,  # hab = [1397.38, 1397.38] for 10% of resilience
res = "1397"

strike = S_0
if strike > S_0:
    moneyness = "OTM"
elif strike < S_0:
    moneyness = "ITM"

#Adress of the AI
ai_is_here = r"/home/clarence/UQAM/Research/Global_Hedging_LOB_projet_Fred_Alex/Code_Alexandre/Pycharm_version/Error_fix_random_init_value/%s/models.ckpt"
# Parameters associated to neural network
nbs_layers = 2
nbs_units = 52  # fixe pour toutes les couches ("dimension" des matrices).
prepro_stock = 'Log-moneyness'
nbs_shares = 1
epochs = 100
train_paths = 400000# used by Alex --> 400000  # juste pour entraînement
test_paths = 10000 # used by Alex -->50000  # juste pour évaluation: out-of-sample performance/résultats pour papier.
lr = 0.001
loss_type = 'RSMSE'  # {'RMSE';'RSMSE'};

if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, 1)
    #V_0 = V_0 + 0.5*V_0 #To check if BS initial value is too high
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, -1)
    #V_0 = V_0 - 0.5*V_0 #To chex if BS initial value is too low




alpha = 1.0 + 8 / 1000.0
beta = 1.0 - 8 / 1000.0

####################################
# Training AI with fixed init values
####################################

name = 'fix_init'

print(name)

# 1) Create tensorflow graph
# - Since its a Python Class, the method __init__ is called upon when the class is created
# - Here, the class contains the tensorflow graph
model_train = DeepAgent_class_fix_init.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, hab, name=name)

# 2) Start training
# - Here, the tensorflow graph has already been initialized
# - When you run a session, it is for this specific tensorflow graph
print('---Training start---')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize all variables from the graph

    # 1) train neural network and output loss per epoch
    _ = model_train._execute_graph_batchwise(train_paths, sess, epochs)



####################################
# Hedging with fixed init values AI
####################################


name = 'fix_init'
model_predict = DeepAgent_class_fix_init.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, hab, name=name)



with tf.Session() as sess:
    # Load trained model
    model_predict.restore(sess,
                          r"/home/clarence/UQAM/Research/Global_Hedging_LOB_projet_Fred_Alex/Code_Alexandre/Pycharm_version/Error_fix_random_init_value/%s/models.ckpt" % name)
    #
    # Compute hedging statistics out-of-sampled
    test_paths = 10000
    deltas_HR, hedging_err_HR, S_t_HR, V_t_HR, A_t_HR, B_t_HR, input_t_HR = model_predict.predict(test_paths, sess, epochs)
    #
    # print stats of hedging err of neural network
    print(" ----------------- ")
    print(" Global hedging %s results" % (loss_type))
    print(" ----------------- ")
    Utils_general.print_stats(hedging_err_HR, deltas_HR, loss_type, "Global hedging - %s" % (loss_type), V_0)

#####################################
# Training AI with random init values
#####################################

name = 'random_init'
print(name)

# 1) Create tensorflow graph
# - Since its a Python Class, the method __init__ is called upon when the class is created
# - Here, the class contains the tensorflow graph
model_train = DeepAgent_class_rand_init.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, hab, name=name)

# 2) Start training
# - Here, the tensorflow graph has already been initialized
# - When you run a session, it is for this specific tensorflow graph
print('---Training start---')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize all variables from the graph

    # 1) train neural network and output loss per epoch
    _ = model_train._execute_graph_batchwise(train_paths, sess, epochs)



#####################################
# Hedging with random init values AI
#####################################

name = 'random_init'
model_predict = DeepAgent_class_rand_init.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, hab, name=name)


with tf.Session() as sess:
    # Load trained model
    model_predict.restore(sess,
                          r"/home/clarence/UQAM/Research/Global_Hedging_LOB_projet_Fred_Alex/Code_Alexandre/Pycharm_version/Error_fix_random_init_value/%s/models.ckpt" % name)
    #
    # Compute hedging statistics out-of-sample
    test_paths = 10000
    deltas_HR, hedging_err_HR, S_t_HR, V_t_HR, A_t_HR, B_t_HR, input_t_HR = model_predict.predict(test_paths, sess, epochs)
    #
    # print stats of hedging err of neural network
    print(" ----------------- ")
    print(" Global hedging %s results" % (loss_type))
    print(" ----------------- ")
    Utils_general.print_stats(hedging_err_HR, deltas_HR, loss_type, "Global hedging - %s" % (loss_type), V_0)


