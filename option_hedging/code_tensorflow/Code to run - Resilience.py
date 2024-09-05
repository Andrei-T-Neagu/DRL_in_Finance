import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime as dt
import numpy as np
#import tensorflow as tf             # TENSORFLOW ---> deep learning (installation CPU)
import tensorflow.compat.v1 as tf            # to keep using Alex's code from tensorflow v1
import matplotlib.pyplot as plt
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
import DeepAgent_class_res_A0_B0_0

tf.compat.v1.disable_eager_execution() #Error with the use of the loss function If not disabled

importlib.reload(Utils_general)
importlib.reload(DeepAgent_class_res_A0_B0_0)

# General got every case
T = 252 / 252  # 252 trading days per year
batch_size = 256  # {128;256;500;1000}
r_borrow = 0.00
r_lend = 0.00
S_0 = 1000.0
hab = [-1, -11]
option_type = 'Call'
position_type = 'Short'  # {'Short', 'Long'}

freq_dyn = 'daily'
freq_rebal = 'monthly'
if (freq_rebal == "weekly"):
    nbs_point_traj = int(T * 52) + 1
elif (freq_rebal == "daily"):
    nbs_point_traj = int(T * 252) + 1
elif (freq_rebal == "monthly"):
    nbs_point_traj = int(T * 12) + 1
elif freq_rebal == "hourly":
    nbs_point_traj = int(T * 252 * 8) + 1 # We suppose stock markets are open 8 hours a day

strike = S_0
if strike > S_0:
    moneyness = "OTM"
elif strike < S_0:
    moneyness = "ITM"
else:
    moneyness = "ATM"

# Parameters associated to neural network
nbs_layers = 2
nbs_units = 64  # fixe pour toutes les couches ("dimension" des matrices).
prepro_stock = 'Log-moneyness'
nbs_shares = 1
epochs = 20
train_paths = 50000# used by Alex --> 400000  # juste pour entraînement
test_paths = 10000 # used by Alex -->50000  # juste pour évaluation: out-of-sample performance/résultats pour papier.
lr = 0.0001
loss_type = 'RSMSE'  # {'RMSE';'RSMSE'};

# model parameters
stock_dyn = 'BSM'
[mu, sigma] = [0.10, 0.1898]
params_vect = [mu, sigma]
if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, -1)


alpha = 1.0
beta = 1.0
res = "none"
name = '%s_%s_Mat=%d_days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d_res=%s' % (
stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)

print(name)

##########################################################
# Training
##########################################################

# 1) Create tensorflow graph
# - Since its a Python Class, the method __init__ is called upon when the class is created
# - Here, the class contains the tensorflow graph
model_train = DeepAgent_class_res_A0_B0_0.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
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



############################
# Computing hedging strategies
###############################

name = '%s_%s_Mat=%d_days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d_res=%s' % (
    stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)
#
#
model_predict = DeepAgent_class_res_A0_B0_0.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, hab, name=name)

with tf.Session() as sess:
    # Load trained model
    model_predict.restore(sess, "C:\\Users\\andrei\\Documents\\school\\grad\\y2\\code_tensorflow\\" + name)
    #
    # Compute hedging statistics out-of-sampled
    test_paths = 10000
    deltas_HR, hedging_err_HR, S_t_HR, V_t_HR, A_t_HR, B_t_HR, input_t_HR = model_predict.predict(test_paths, sess, epochs)
    #
    # print stats of hedging err of neural network
    print(" ----------------- ")
    print(" Global hedging %s results" % (loss_type))
    print(" ----------------- ")
    Utils_general.print_stats(hedging_err_HR, deltas_HR, loss_type, "Deep hedge - %s" % (loss_type), V_0)
    #
    # def delta_hedge_res(St_traj, r_borrow, r_lend, sigma, T, alpha, beta, option_type, position_type, strike, V_0, nbs_shares, hab):
    # 4) Delta-hedging statistics
    print(" ----------------- ")
    print(" Delta hedge results")
    print(" ----------------- ")
    deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t_HR, r_borrow, r_lend, sigma, T, alpha, beta, option_type, position_type, strike, V_0, nbs_shares, hab)
    Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)

    # sess.close()
################################################################
# Point prediction
################################################################


name = '%s_%s_Mat=%d_days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d_res=%s' % (
    stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)
#
#
model_predict = DeepAgent_class_res_A0_B0_0.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, hab, name=name)

with tf.Session() as sess:
    # Load trained model
    model_predict.restore(sess, "C:\\Users\\andrei\\Documents\\school\\grad\\y2\\code_tensorflow\\" + name)
    t_t = 11.0
    Vt = 0.0
    St = 1300.0
    At = 0.0
    Bt = 0.0
    deltat = 0.0
    delta_pred = model_predict.point_pred(sess, t_t, St, Vt, At, Bt, deltat)
    print(delta_pred)
#
#
# #########################################################
# #Case  # 2, Low resilience, ha=hb=10, TRAINING
# #########################################################
# hab = [10.0, 10.0]
# alpha = 1.02
# beta = 0.98
# res = "ha=hb=10"
# name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d.res=%s' % (
# stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)
#
# print(name)
#
#
# # 1) Create tensorflow graph
# # - Since its a Python Class, the method __init__ is called upon when the class is created
# # - Here, the class contains the tensorflow graph
# model_train = DeepAgent_class_res.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
#                                               S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
#                                               nbs_layers, nbs_units, lr, prepro_stock,
#                                               nbs_shares, freq_dyn, hab, name=name)
#
# # 2) Start training
# # - Here, the tensorflow graph has already been initialized
# # - When you run a session, it is for this specific tensorflow graph
# print('---Training start---')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())  # Initialize all variables from the graph
#
#     # 1) train neural network and output loss per epoch
#     _ = model_train._execute_graph_batchwise(train_paths, sess, epochs)
#
#
# ###################################################
# # END OF TRAINING
# ###################################################
#
# #################################################
# # Compute hedging strategy for high resilience
    # #################################################
#
# #################################################
# # Compute hedging strategy for low resilience
# #################################################
# hab = [10.0, 10.0]
# alpha = 1.01
# beta = 0.99
# res = "ha=hb=10"
# name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d.res=%s' % (
# stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)
#
#
# model_predict = DeepAgent_class_res.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
#                                                 S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
#                                                 nbs_layers, nbs_units, lr, prepro_stock,
#                                                 nbs_shares, freq_dyn, hab, name=name)
#
#
# tf.compat.v1.random.set_random_seed(10)
#
# with tf.Session() as sess:
#     # Load trained model
#     model_predict.restore(sess,
#                           r"/home/clarence/UQAM/Research/Global_Hedging_LOB_projet_Fred_Alex/Code_Alexandre/Pycharm_version/Models_w_res/%s/models.ckpt" % name)
#
#     # Compute hedging statistics out-of-sample
#     deltas_LR, hedging_err_LR, S_t_LR, V_t_LR, A_t_LR, B_t_LR, input_t_LR = model_predict.predict(test_paths, sess,
#                                                                                                       epochs)
#
#     # print stats of hedging err of neural network
#     print(" ----------------- ")
#     print(" Global hedging %s results" % (loss_type))
#     print(" ----------------- ")
#     Utils_general.print_stats(hedging_err_LR, deltas_LR, loss_type, "Global hedging - %s" % (loss_type), V_0)
#
#
# # 4) Delta-hedging statistics
# print(" ----------------- ")
# print(" Delta hedge results")
# print(" ----------------- ")
# deltas_DH, hedging_err_DH = Utils_general.delta_hedge(nbs_point_traj, test_paths, r_borrow, r_lend, mu, sigma,
#                                                       S_0, T, alpha, beta, option_type, position_type, strike, V_0,
#                                                       nbs_shares)
# _, _ = Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta-hedge", "Delta-hedge", V_0)
#
#
