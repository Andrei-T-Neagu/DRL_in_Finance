
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
import DeepAgent_class_rand_init
import utils_cs

tf.compat.v1.disable_eager_execution() #Error with the use of the loss function If not disabled

importlib.reload(Utils_general)
importlib.reload(DeepAgent_class_rand_init)

##################
# Inputs
##################

# General got every case
T = 1 / 252  # 252 trading days per year
batch_size = 20000  # {128;256;500;1000}
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
ai_is_here = r"/home/clarence/UQAM/Research/Global_Hedging_LOB_projet_Fred_Alex/Code_Alexandre/Pycharm_version/Study_nn_function/%s/models.ckpt"
# Parameters associated to neural network
nbs_layers = 2
nbs_units = 52  # fixe pour toutes les couches ("dimension" des matrices).
prepro_stock = 'Log-moneyness'
nbs_shares = 1
epochs = 50
train_paths = 600000# used by Alex --> 400000  # juste pour entraînement
test_paths = 200000 # used by Alex -->50000  # juste pour évaluation: out-of-sample performance/résultats pour papier.
lr = 0.001
loss_type = 'RSMSE'  # {'RMSE';'RSMSE'};

if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, 1)
    #V_0 = V_0 + 0.5*V_0 #To check if BS initial value is too high
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, -1)
    #V_0 = V_0 - 0.5*V_0 #To chex if BS initial value is too low



#############################################################################
# Effect of At on the strategy when portfolio value is Black Scholes value.
#############################################################################

#######WEAK

nb_models = 5  # Different values of alpha, beta
nb_At = 100 # Number of time steps studied

t_exp = np.zeros((nb_At, nb_models))
cost_S0 = np.zeros((nb_At, nb_models))
# Matrices for the results [time, models ]
# Loop on time
rmse = []
std = []
for i in [0,1,4]:
    print(i)
    alpha = 1.0 + i * 2/ 1000.0
    beta = 1.0 - i * 2/ 1000.0

    name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d.res=%s' % (stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)

    model_predict = DeepAgent_class_rand_init.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                                  S_0, T, alpha, beta, loss_type, option_type, position_type, strike,
                                                  V_0,
                                                  nbs_layers, nbs_units, lr, prepro_stock,
                                                  nbs_shares, freq_dyn, hab, name=name)

    with tf.Session() as sess:
        # Load trained model

        model_predict.restore(sess, ai_is_here % name)

        S_0 = 100.0
        At = np.linspace(0,1.5,nb_At)
        Bt = 0.0
        deltat = 0.0
        # Loop on time
        for j_index, j in enumerate(At):
            Vt = Utils_general.BlackScholes_price(S_0, T , r_borrow, sigma, strike, 1)
            t_exp[j_index, i] = model_predict.point_pred(sess, 0, S_0, Vt, j, Bt, deltat)
            cost_S0[j_index, i] = Utils_general.cost_buying(S_0, 1, alpha, j)

        deltas_HR, hedging_err_HR, S_t_HR, V_t_HR, A_t_HR, B_t_HR, input_t_HR = model_predict.predict(test_paths, sess, epochs)
        print(" ----------------- ")
        print(" Global hedging %s results" % (loss_type))
        print(" ----------------- ")
        _,_, tmp_rmse, tmp_std = Utils_general.print_stats(hedging_err_HR, deltas_HR, loss_type, "Global hedging - %s" % (loss_type), V_0)
        rmse.append(tmp_rmse)
        std.append(tmp_std)


weak_res_At = t_exp[:,[0,1,4]]
rmse_weak_res_At = rmse
std_weak_res_At = std


#######MED

importlib.reload(DeepAgent_class_rand_init)
hab = [4642.01, 4642.01] #[9284.023, 9284.023] # for 99% of resilience #[4642.01, 4642.01]   for 50%,  # hab = [1397.38, 1397.38] for 10% of resilience
res = "4642"


nb_models = 5  # Different values of alpha, beta
nb_At = 100 # Number of time steps studied

t_exp = np.zeros((nb_At, nb_models))
cost_S0 = np.zeros((nb_At, nb_models))
# Matrices for the results [time, models ]
# Loop on time
rmse = []
std = []
for i in [0,1,4]:
    print(i)
    alpha = 1.0 + i * 2/ 1000.0
    beta = 1.0 - i * 2/ 1000.0

    name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d.res=%s' % (stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)

    model_predict = DeepAgent_class_rand_init.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                                  S_0, T, alpha, beta, loss_type, option_type, position_type, strike,
                                                  V_0,
                                                  nbs_layers, nbs_units, lr, prepro_stock,
                                                  nbs_shares, freq_dyn, hab, name=name)

    with tf.Session() as sess:
        # Load trained model

        model_predict.restore(sess, ai_is_here % name)

        S_0 = 100.0
        At = np.linspace(0,1.5,nb_At)
        Bt = 0.0
        deltat = 0.0
        # Loop on time
        for j_index, j in enumerate(At):
            Vt = Utils_general.BlackScholes_price(S_0, T , r_borrow, sigma, strike, 1)
            t_exp[j_index, i] = model_predict.point_pred(sess, 0, S_0, Vt, j, Bt, deltat)
            cost_S0[j_index, i] = Utils_general.cost_buying(S_0, 1, alpha, j)

        deltas_HR, hedging_err_HR, S_t_HR, V_t_HR, A_t_HR, B_t_HR, input_t_HR = model_predict.predict(test_paths, sess, epochs)
        print(" ----------------- ")
        print(" Global hedging %s results" % (loss_type))
        print(" ----------------- ")
        _,_, tmp_rmse, tmp_std = Utils_general.print_stats(hedging_err_HR, deltas_HR, loss_type, "Global hedging - %s" % (loss_type), V_0)
        rmse.append(tmp_rmse)
        std.append(tmp_std)


med_res_At = t_exp[:,[0,1,4]]
rmse_med_res_At = rmse
std_med_res_At = std


#######STR


importlib.reload(DeepAgent_class_rand_init)
hab = [9284.023, 9284.023] #[9284.023, 9284.023] # for 99% of resilience #[4642.01, 4642.01]   for 50%,  # hab = [1397.38, 1397.38] for 10% of resilience
res = "9284"

nb_models = 5  # Different values of alpha, beta
nb_At = 100 # Number of time steps studied

t_exp = np.zeros((nb_At, nb_models))
cost_S0 = np.zeros((nb_At, nb_models))
# Matrices for the results [time, models ]
# Loop on time
rmse = []
std = []
for i in [0,1,4]:
    print(i)
    alpha = 1.0 + i * 2/ 1000.0
    beta = 1.0 - i * 2/ 1000.0

    name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d.res=%s' % (stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)

    model_predict = DeepAgent_class_rand_init.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                                  S_0, T, alpha, beta, loss_type, option_type, position_type, strike,
                                                  V_0,
                                                  nbs_layers, nbs_units, lr, prepro_stock,
                                                  nbs_shares, freq_dyn, hab, name=name)

    with tf.Session() as sess:
        # Load trained model

        model_predict.restore(sess, ai_is_here % name)

        S_0 = 100.0
        At = np.linspace(0,1.5,nb_At)
        Bt = 0.0
        deltat = 0.0
        # Loop on time
        for j_index, j in enumerate(At):
            Vt = Utils_general.BlackScholes_price(S_0, T , r_borrow, sigma, strike, 1)
            t_exp[j_index, i] = model_predict.point_pred(sess, 0, S_0, Vt, j, Bt, deltat)
            cost_S0[j_index, i] = Utils_general.cost_buying(S_0, 1, alpha, j)

        deltas_HR, hedging_err_HR, S_t_HR, V_t_HR, A_t_HR, B_t_HR, input_t_HR = model_predict.predict(test_paths, sess, epochs)
        print(" ----------------- ")
        print(" Global hedging %s results" % (loss_type))
        print(" ----------------- ")
        _,_, tmp_rmse, tmp_std = Utils_general.print_stats(hedging_err_HR, deltas_HR, loss_type, "Global hedging - %s" % (loss_type), V_0)
        rmse.append(tmp_rmse)
        std.append(tmp_std)


str_res_At = t_exp[:,[0,1,4]]
rmse_str_res_At = rmse
std_str_res_At = std


################At############
At = np.linspace(0, 1.5, nb_At)
plt.plot(At, weak_res_At[:,0], color='black', label='Perfectly liquid')
plt.plot(At, weak_res_At[:,1], color='green', label='weak, deep')
plt.plot(At, weak_res_At[:,2], color='green', linestyle='--', label='weak res, shallow')
plt.plot(At, med_res_At[:,1], color='blue', label='medium, deep')
plt.plot(At, med_res_At[:,2], color='blue', linestyle='--', label='medium, shallow')
plt.plot(At, str_res_At[:,1], color='red', label='strong, deep')
plt.plot(At, str_res_At[:,2], color='red', linestyle='--', label='strong, shallow')
plt.xlabel('At')
plt.ylabel('Number of shares')
plt.legend(loc='best')



##############
#import pickle

#with open('effect_of_At_error', 'wb') as f:
#    pickle.dump([rmse_weak_res_At, rmse_med_res_At, rmse_str_res_At],f)
#    pickle.dump([std_weak_res_At, std_med_res_At, std_str_res_At],f)
