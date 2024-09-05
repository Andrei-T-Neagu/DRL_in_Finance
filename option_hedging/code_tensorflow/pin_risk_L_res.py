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
import utils_cs

tf.compat.v1.disable_eager_execution() #Error with the use of the loss function If not disabled

importlib.reload(Utils_general)
importlib.reload(DeepAgent_class_res)

##################
#Example of inputs
##################

# General got every case
T = 1 / 252  # 252 trading days per year
batch_size = 3000  # {128;256;500;1000}
r_borrow = 0.00
r_lend = 0.00
S_0 = 1000.0
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
[mu, sigma] = [0.10, 0.1898]
params_vect = [mu, sigma]
hab = [10.0, 10.0]

strike = S_0
if strike > S_0:
    moneyness = "OTM"
elif strike < S_0:
    moneyness = "ITM"


# Parameters associated to neural network
nbs_layers = 2
nbs_units = 52  # fixe pour toutes les couches ("dimension" des matrices).
prepro_stock = 'Log-moneyness'
nbs_shares = 1
epochs = 100
train_paths = 200000# used by Alex --> 400000  # juste pour entraînement
test_paths = 10000 # used by Alex -->50000  # juste pour évaluation: out-of-sample performance/résultats pour papier.
lr = 0.0001
loss_type = 'RSMSE'  # {'RMSE';'RSMSE'};

if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, 1)
    #V_0 = V_0 + 0.5*V_0 #To check if BS initial value is too high
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, sigma, strike, -1)
    #V_0 = V_0 - 0.5*V_0 #To chex if BS initial value is too low



#########################################################
#Training and creation of AI for decreasing steepness of LOB (increase in liquidity cost).
#########################################################

for i in range(0, 5):
    alpha = 1.0 + i * 2 / 1000.0
    beta = 1.0 - i * 2 / 1000.0
    res = "ha=hb=10"
    name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d.res=%s' % (
stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)

    print(name)

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



##############################################
# Computation of hedging strategies and statistics
##############################################
##
# Don't forget to set the seed in the class and reload the class
##

##

importlib.reload(DeepAgent_class_res_A0_B0_0)

turnover_series_tens = np.array([])
turnover_series_DH = np.array([])
deltas_series = np.array([])
for i in range(0, 5):
    print(i)
    alpha = 1.0 + i * 2 / 1000.0
    beta = 1.0 - i * 2 / 1000.0
    res = "ha=hb=10"
    name = '%s_%s_Mat=%d days_%s_%s_alpha_%.4f_beta_%.4f_mu_%.4f_sigma_%.4f_strike_%.1f_shares=%d.res=%s' % (stock_dyn, option_type, int(T * 252), freq_rebal, loss_type, alpha, beta, mu, sigma, strike, nbs_shares, res)

    model_predict = DeepAgent_class_res_A0_B0_0.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect,
                                                  S_0, T, alpha, beta, loss_type, option_type, position_type, strike,
                                                  V_0,
                                                  nbs_layers, nbs_units, lr, prepro_stock,
                                                  nbs_shares, freq_dyn, hab, name=name)

    with tf.Session() as sess:
        # Load trained model

        model_predict.restore(sess,
                              r"/home/clarence/UQAM/Research/Global_Hedging_LOB_projet_Fred_Alex/Code_Alexandre/Pycharm_version/Models_w_res/%s/models.ckpt" % name)

        # Compute hedging statistics out-of-sample
        deltas_tensor, hedging_err_tensor, S_t_tensor, V_t_tensor, A_t_tensor, B_t_tensor, input_t_tensor = model_predict.predict(test_paths, sess, epochs)

        deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t_tensor, r_borrow, r_lend, sigma, T, alpha, beta, option_type, position_type, strike, V_0, nbs_shares, hab)
       # print(np.mean(hedging_err_tensor))


        if i == 0:
            deltas_series = np.append(deltas_series, deltas_tensor[:, 0])
            turnover_series_tens = np.append(turnover_series_tens, utils_cs.avg_turn_series(deltas_tensor))
            turnover_series_DH = np.append(turnover_series_DH, utils_cs.avg_turn_series(deltas_DH))
        else:
            deltas_series = np.vstack((deltas_series, np.transpose(deltas_tensor[:, 0])))
            turnover_series_tens = np.vstack([turnover_series_tens, utils_cs.avg_turn_series(deltas_tensor)])
            turnover_series_DH = np.vstack([turnover_series_DH, utils_cs.avg_turn_series(deltas_DH)])
        # print stats of hedging err of neural network
        print(" ----------------- ")
        print(" Global hedging %s results" % (loss_type))
        print(" ----------------- ")
    #    Utils_general.print_stats(hedging_err_tensor, deltas_tensor, loss_type, "Global hedging - %s" % (loss_type), V_0)
        #
        # def delta_hedge_res(St_traj, r_borrow, r_lend, sigma, T, alpha, beta, option_type, position_type, strike, V_0, nbs_shares, hab):
        # 4) Delta-hedging statistics
        print(" ----------------- ")
        print(" Delta hedge results")
        print(" ----------------- ")
     #   deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t_tensor, r_borrow, r_lend, sigma, T, alpha, beta,
#                                                                  option_type, position_type, strike, V_0, nbs_shares, hab)
     #   _, _ = Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta-hedge", "Delta-hedge", V_0)


    ###########################
     # deltas vs At Bt St
     ###########################
    # traj_index = 0
    # plt.rcParams['text.usetex'] = True
    # fig, ax = plt.subplots(4)
    # ax[0].plot(deltas_tensor[:,traj_index], label='Shares')
    # ax[0].legend()
    # ax[1].plot(A_t_tensor[:, traj_index], label='$A_t$')
    # ax[1].legend()
    # ax[2].plot(B_t_tensor[:, traj_index], label='$B_t$')
    # ax[2].legend()
    #
    # ax[3].plot(S_t_tensor[:,traj_index], label='$S_t$')
    # ax[3].legend()

deltas_series = np.transpose(deltas_series)

fig, ax = plt.subplots(5)
fig.suptitle('Pin risk, low resilience')
t = np.array(range(len(S_t_tensor[:, 0])))
for i in range(4):
    ax[i].set_xlim([-0.2, 8.2])
    ax[i].plot(deltas_series[:, i])
    legend = r' $\alpha$ , $\beta$ = 1  $\pm$ %1.3f' % (i * 2.0 / 1000.0)
    ax[i].text(7, 0.4, legend)
    ax[i].set_ylabel('nb of shares')

ax[4].set_xlabel('time')
ax[4].set_xlim([-0.2, 8.2])
ax[4].plot(t, S_t_tensor[:, 0])
legend = r'GBM'
ax[4].set_ylabel('GBM')

legend = [r'$\alpha, \beta$ = 1 $\pm$ %1.3f' % (i * 2.0 / 1000.0) for i in range(5)]
plt.plot(np.transpose(turnover_series_tens))
plt.legend(legend)
plt.title('Turnover series, low resilience')
plt.xlabel('time')
plt.ylabel('turnover')

plt.figure()
plt.plot(turnover_series_tens[0,:], label='high liquidity')
plt.plot(turnover_series_tens[1,:], label='medium liquidity')
plt.plot(turnover_series_tens[2,:], label='low liquidity')
plt.xlabel('Time (hours)')
plt.ylabel('Avg abs rebal')
plt.legend()

