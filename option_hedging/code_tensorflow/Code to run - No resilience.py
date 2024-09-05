import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime as dt
import numpy as np
#import tensorflow as tf             # TENSORFLOW ---> deep learning (installation CPU)
import tensorflow.compat.v1 as tf            # to keep using Alex's code from tensorflow v1
import matplotlib.pyplot as plt
import random
#from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.optimizers import Adam


#from keras import backend as K
from tensorflow.compat.v1.keras import backend as K       # to keep using Alex's code from tensorflow v1

from scipy import stats
from scipy.special import ndtri
#import import_ipynb
import importlib

import Utils_general
import DeepAgent_Class_nores

tf.compat.v1.disable_eager_execution() #Error with the use of the loss function If not disabled

importlib.reload(Utils_general)
importlib.reload(DeepAgent_Class_nores)

##################
#Example of inputs
##################

# General got every case
T = 30 / 252  # 252 trading days per year
batch_size = 50  # {128;256;500;1000}
r_borrow = 0.00
r_lend = 0.00
S_0 = 1000.0
option_type = 'Call'
position_type = 'Short'  # {'Short', 'Long'}
freq_dyn = 'daily'
freq_rebal = 'daily'
if (freq_rebal == "weekly"):
    nbs_point_traj = int(T * 52) + 1
elif (freq_rebal == "daily"):
    nbs_point_traj = int(T * 252) + 1
elif (freq_rebal == "monthly"):
    nbs_point_traj = int(T * 12) + 1
strike = S_0
if strike > S_0:
    moneyness = "OTM"
elif strike < S_0:
    moneyness = "ITM"
else:
    moneyness = "ATM"

# Parameters associated to neural network
nbs_layers = 2
nbs_units = 52  # fixe pour toutes les couches ("dimension" des matrices).
prepro_stock = 'Log-moneyness'
nbs_shares = 1
epochs = 20
train_paths = 50000# used by Alex --> 400000  # juste pour entraînement
test_paths = 10000 # used by Alex -->50000  # juste pour évaluation: out-of-sample performance/résultats pour papier.
lr = 0.001
loss_type = 'RMSE'  # {'RMSE';'RSMSE'};

# model parameters
stock_dyn = 'BSM'
[mu, sigma] = [0.10, 0.1898]
params_vect = [mu, sigma]
if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, -1)

##############################################
#Case  # 1 - no impact (alpha = beta = 1.00)
##############################################
alpha = 1.00
beta = 1.00

# name ---> le nom du fichier utilisé pour sauvegarder les paramètres du NNet une fois entraîner.
name = '%s_%s_Mat=%d_days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d' % (
stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares)
print(name)
BSM_Call_Mat = 30
name = '%s_%s_Mat=%d_days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d' % (
stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares)
print(name)


#Next box: train NNet( and saves the best set of parameters). evaluate performance on test set: load the model parameters
#present hedging statistics on test set with the trained NNet and with Delta-hedging strategy (see final output).

# 1) Create tensorflow graph
# - classe dans le script "DeepAgent_Class_nores.ipynb";
model_train = DeepAgent_Class_nores.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, name=name)

# 2) Start training
print('---Training start---')
with tf.Session() as sess:  # tf ---> tensorflow (librairie soit GPU ou CPU);
    sess.run(tf.global_variables_initializer())  # Initialize all variables from the graph

    # 1) train neural network and output loss per epoch
    # - _execute_graph_batchwise ---> "DeepAgent_Class_nores.ipynb"
    # - saving: done within this function.
    _ = model_train._execute_graph_batchwise(train_paths, sess, epochs)

    sess.close()
# ------------------------- #
# TRAINING IS DONE
# - best parameters are saved;
# ------------------------- #

# ----------------------------------------------------#
# Maintenant: "évaluation de police" sur le test set.
# - pour faire l'évaluation ---> seulement utiliser le bout de code en dessous
# ----------------------------------------------------- #
# 3) Compute hedging strategy
# A) créer la classe
# - créer la classe
model_predict = DeepAgent_Class_nores.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                                S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                                nbs_layers, nbs_units, lr, prepro_stock,
                                                nbs_shares, freq_dyn, name=name)

with tf.Session() as sess:
    # Load trained model
    # - restore ---> load les paramètres pour Nnet;
    model_predict.restore(sess, "C:\\Users\\andrei\\Documents\\school\\grad\\y2\\code_tensorflow\\" + name)

    # Compute hedging statistics out-of-sample
    # - deltas_tensor      ---> tensor de positions pour toutes les trajectoires X pas de temps;
    #   - position en shares;
    # - hedging_err_tensor ---> vector des hedging errors finales pour chaque trajectoire;
    #   - (Phi - V_T) ---> FORMULE DE VALEUR DANS L'ARTICLE AVANT APPLICATION {MSE;SMSE;ETC.}
    # - S_t_tensor         ---> prix du sous-jacent à chaque trajectoire X pas de temps;
    # - V_t                ---> valeur du portefeuille (avec modèle d'impact/liquidation);
    # - input_t_tensor     ---> X_t = [S_t;V_t;T-t;...] par trajectoire X pas de temps.
    deltas_tensor, hedging_err_tensor, S_t_tensor, V_t_tensor, input_t_tensor = model_predict.predict(test_paths, sess,
                                                                                                      epochs)

    # print stats of hedging err of neural network
    print(" ----------------- ")
    print(" Global hedging %s results" % (loss_type))
    print(" ----------------- ")
    Utils_general.print_stats(hedging_err_tensor, deltas_tensor, loss_type, "Global hedging - %s" % (loss_type), V_0)

    sess.close()
# 4) Delta-hedging statistics
# ----------------------------------------------------#
# Maintenant: "évaluation de police" sur le test set.
# - pour faire l'évaluation ---> seulement utiliser le bout de code en dessous
# ----------------------------------------------------- #
# 3) Compute hedging strategy
# A) créer la classe
# - créer la classe
model_predict = DeepAgent_Class_nores.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                                S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                                nbs_layers, nbs_units, lr, prepro_stock,
                                                nbs_shares, freq_dyn, name=name)

with tf.Session() as sess:
    # Load trained model
    # - restore ---> load les paramètres pour Nnet;
    model_predict.restore(sess, "C:\\Users\\andrei\\Documents\\school\\grad\\y2\\code_tensorflow\\" + name)

    # Compute hedging statistics out-of-sample
    # - deltas_tensor      ---> tensor de positions pour toutes les trajectoires X pas de temps;
    #   - position en shares;
    # - hedging_err_tensor ---> vector des hedging errors finales pour chaque trajectoire;
    #   - (Phi - V_T) ---> FORMULE DE VALEUR DANS L'ARTICLE AVANT APPLICATION {MSE;SMSE;ETC.}
    # - S_t_tensor         ---> prix du sous-jacent à chaque trajectoire X pas de temps;
    # - V_t                ---> valeur du portefeuille (avec modèle d'impact/liquidation);
    # - input_t_tensor     ---> X_t = [S_t;V_t;T-t;...] par trajectoire X pas de temps.
    deltas_tensor, hedging_err_tensor, S_t_tensor, V_t_tensor, input_t_tensor = model_predict.predict(test_paths, sess,
                                                                                                      epochs)

    # print stats of hedging err of neural network
    print(" ----------------- ")
    print(" Global hedging %s results" % (loss_type))
    print(" ----------------- ")
    Utils_general.print_stats(hedging_err_tensor, deltas_tensor, loss_type, "Global hedging - %s" % (loss_type), V_0)

# 4) Delta-hedging statistics
print(" ----------------- ")
print(" Delta hedge results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge(nbs_point_traj, test_paths, r_borrow, r_lend, mu, sigma,
                                                      S_0, T, alpha, beta, option_type, position_type, strike, V_0,
                                                      nbs_shares)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta-hedge", "Delta-hedge", V_0)

print(" ----------------- ")
print(" Delta hedge results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge(nbs_point_traj, test_paths, r_borrow, r_lend, mu, sigma,
                                                      S_0, T, alpha, beta, option_type, position_type, strike, V_0,
                                                      nbs_shares)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta-hedge", "Delta-hedge", V_0)


#########################################################
#Case  # 2 - with impact ([alpha, beta] = [1.02, 0.98])
#########################################################

alpha = 1.002
beta = 0.998
name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d' % (
stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares)
print(name)



# 1) Create tensorflow graph
# - Since its a Python Class, the method __init__ is called upon when the class is created
# - Here, the class contains the tensorflow graph
model_train = DeepAgent_Class_nores.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                              S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                              nbs_layers, nbs_units, lr, prepro_stock,
                                              nbs_shares, freq_dyn, name=name)

# 2) Start training
# - Here, the tensorflow graph has already been initialized
# - When you run a session, it is for this specific tensorflow graph
print('---Training start---')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize all variables from the graph

    # 1) train neural network and output loss per epoch
    _ = model_train._execute_graph_batchwise(train_paths, sess, epochs)


sess.close()

model_predict = DeepAgent_Class_nores.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                                S_0, T, alpha, beta, loss_type, option_type, position_type, strike, V_0,
                                                nbs_layers, nbs_units, lr, prepro_stock,
                                                nbs_shares, freq_dyn, name=name)


with tf.Session() as sess:
    # Load trained model
    model_predict.restore(sess, "C:\\Users\\andrei\\Documents\\school\\grad\\y2\\code_tensorflow\\" + name)

    # Compute hedging statistics out-of-sample
    deltas_tensor, hedging_err_tensor, S_t_tensor, V_t_tensor, input_t_tensor = model_predict.predict(test_paths, sess,
                                                                                                      epochs)

    # print stats of hedging err of neural network
    print(" ----------------- ")
    print(" Global hedging %s results" % (loss_type))
    print(" ----------------- ")
    Utils_general.print_stats(hedging_err_tensor, deltas_tensor, loss_type, "Global hedging - %s" % (loss_type), V_0)

sess.close()
# 4) Delta-hedging statistics
print(" ----------------- ")
print(" Delta hedge results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge(nbs_point_traj, test_paths, r_borrow, r_lend, mu, sigma,
                                                      S_0, T, alpha, beta, option_type, position_type, strike, V_0,
                                                      nbs_shares)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta-hedge", "Delta-hedge", V_0)