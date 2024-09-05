import datetime as dt
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf          # to keep using Alex's code from tensorflow v1
import matplotlib.pyplot as plt
import random
#from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.optimizers import Adam   # to keep using Alex's code from tensorflow v1
from keras import backend as K
from tensorflow.compat.v1.keras import backend as K       # to keep using Alex's code from tensorflow v1
tf.compat.v1.disable_eager_execution() #Error with the use of the loss function If not disabled


class DeepAgent(object):
    """
    Inputs:
    nbs_point_traj: if [S_0,...,S_N], nbs_point_traj = N+1, i.e. nbs_point_traj = nbs of rebalancement + 1
    batch_size    : size of the batch
    r_borrow      : rate to borrow, should be larger than r_lend
    r_lend        : rate to lend/invest at bank
    stock_dyn     : {"BSM"}
    params_vect   : vector of parameters for dynamics of underlying
                    - BSM      : [mu, sigma]
    T             : time-to-maturity in years
    loss_type     : loss applied on terminal reward{CVaR, MSE, SMSE, VaR}
    option_type   : {Call, Put}
    position_type : {Long, Short}
    nbs_layers    : number of layers
    nbs_units     : fixed number of units per layer
    lr            : learning rate for Adam optimizer
    prepro_stock  : {Log, Log-moneyness, Nothing}. Stock price input to NNet's are norm., but loss computa. are denorm.
    freq_dyn      : frequency of simulated dynamics {"daily"}
    hab           : parameters in the exponential resilience hab = [param for A_t, param for B_t], for no impact set hab == [-1, -1]
    name          : Model name to be saved
    """

    def __init__(self, nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, freq_dyn, hab, name='model'):

        # Clears out the current graph (does not initialize variables values)
        # Does not initialize variable values, this is done once at the beginning with
        # sess.run(tf.global_variables_initializer())
        #Ancienne commande qui n'existe plus dans TF2 -> tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()

        # 0) Just 'self.' inputs of the class
        self.nbs_point_traj = nbs_point_traj
        self.batch_size = batch_size
        self.r_borrow = r_borrow
        self.r_lend = r_lend
        self.stock_dyn = stock_dyn
        self.S_0 = S_0
        self.T = T
        self.alpha = alpha  # value for impact factor when buying
        self.beta = beta  # value for impact factor when selling
        self.loss_type = loss_type
        self.option_type = option_type
        self.position_type = position_type

        self.V_0 = V_0
        self.nbs_layers = nbs_layers  # with MLP, useless currently
        self.nbs_units = nbs_units
        self.lr = lr
        self.prepro_stock = prepro_stock
        self.nbs_shares = nbs_shares
        self.freq_dyn = freq_dyn
        self.delta_t = tf.zeros(self.batch_size)  # number of shares at each time-step
        self.N = self.nbs_point_traj - 1  # number of time-steps
        self.h = self.T / self.N  # time-step size

        self.A_0 = 0 #Initial value of the impact
        self.ha = hab[0] #Resilience parameter, ask part
        self.B_0 = 0 #Initial value of the impact
        self.hb = hab[1] #Resilience dparameter, bid part
        # self.strike = self.S_0*((1 + self.nbs_shares)**self.beta-1)
        self.strike = strike
        ######################################

        print(self.V_0)

        # 1) Extract model parameters
        if (self.stock_dyn == "BSM"):
            self.mu, self.sigma = params_vect

        # 2) Portfolio value prior to trading at each time-step
        # - if long, you buy the option by borrowing V_0 from bank
        if (self.position_type == "Long"):
            V_t = -self.V_0 * tf.ones(self.batch_size)  # [batch_size]

        # - if short, you receive the premium that you put in the bank
        elif (self.position_type == "Short"):
            V_t = self.V_0 * tf.ones(self.batch_size)  # [batch_size]

        # tf.expand_dims add a dimension (tensor of length 1) at axis=0
        self.V_t_tensor = tf.expand_dims(V_t, axis=0)  # [1, batch_size]
        self.S_t_tensor = tf.expand_dims(S_0 * tf.ones(self.batch_size), axis=0)
        self.A_t_tensor = tf.expand_dims(self.A_0 * tf.ones(self.batch_size), axis=0)
        self.B_t_tensor = tf.expand_dims(self.B_0 * tf.ones(self.batch_size), axis=0)

        # 3) Processing stock price
        if (self.prepro_stock == "Log"):
            self.S_t = tf.math.log(S_0) * tf.ones(self.batch_size)
        elif (self.prepro_stock == "Log-moneyness"):
            self.S_t = tf.math.log(S_0 / self.strike) * tf.ones(self.batch_size)
        elif (self.prepro_stock == "None"):
            self.S_t = S_0 * tf.ones(self.batch_size)

        A_t = self.A_0 * tf.ones(self.batch_size)
        B_t = self.B_0 * tf.ones(self.batch_size)

        # 4) Define layers for MLP (single FFNN) and LSTM
        self.layer_1 = tf.compat.v1.layers.Dense(self.nbs_units, tf.nn.relu)
        self.layer_2 = tf.compat.v1.layers.Dense(self.nbs_units, tf.nn.relu)
        self.layer_out = tf.compat.v1.layers.Dense(1, None)

        # 5) Loop over each time-step of the financial market
        for t in range(self.N):
            # 5.1) Construct feature vector at the beginning of time 't'
            # input_t[i, :] = [S_t, delta_t, V_t, A_t] is the basis
            # - S_t and V_t are normalized already
            # input_t.shape = [batch_size, 4]
            if (self.alpha == 1.00 and self.beta == 1.00):  # no impact, don,t include 'delta_t'
                input_t = tf.concat([tf.expand_dims(self.S_t, axis=1), tf.expand_dims(self.delta_t, axis=1),
                                     tf.expand_dims(V_t / self.V_0, axis=1)], axis=1)
            else:
                input_t = tf.concat([tf.expand_dims(self.S_t, axis=1), tf.expand_dims(self.delta_t, axis=1),
                                     tf.expand_dims(V_t / self.V_0, axis=1), tf.expand_dims(A_t, axis=1),
                                     tf.expand_dims(B_t, axis=1)], axis=1)

            # 5.2) de-normalize price
            self.S_t = self.inverse_processing(self.S_t, self.prepro_stock)

            # 5.4) Compute hedge and update cash amount in bank account
            # Need to add time-step (add h*t, the time-step, a normalization)
            input_t = tf.concat([input_t, tf.expand_dims(self.h * t * tf.ones(self.batch_size), axis=1)], axis=1)

            # This is only to output self.input_t_tensor out of function to make sure the input works
            if (t == 0):
                self.input_t_tensor = tf.expand_dims(input_t, axis=0)
            else:
                self.input_t_tensor = tf.concat([self.input_t_tensor, tf.expand_dims(input_t, axis=0)], axis=0)

            layer = self.layer_1(input_t)
            layer = self.layer_2(layer)
            self.delta_t_next = self.layer_out(layer)

            # TEST: testing the results if we only hedge at the last time step
            # if t < self.N-1:
            #     delta_t_next = tf.zeros([self.batch_size, 1])

            # Once the hedge is computed: 1) compile in self.strategy; 2) update Y_t
            if (t == 0):
                self.strategy = tf.expand_dims(self.delta_t_next, axis=0)  # [1, batch_size, 1]
                cashflow = self.liquid_func(self.S_t, -self.delta_t_next[:, 0], A_t, B_t)
                self.Y_t = V_t + cashflow  # time-0 amount in the bank account
                #Compute impact and resilience
                if self.ha == -1:
                    A_t = tf.constant(0, dtype=tf.float32, shape=self.delta_t_next[:, 0].shape)
                else:
                    impact_ask = tf.cast(tf.greater(self.delta_t_next[:, 0], tf.zeros(self.batch_size)), tf.float32) * self.delta_t_next[:, 0]
                    A_t = (A_t + impact_ask) * tf.exp(-self.ha * self.h)  # Update impact with

                    # transaction and include resilience for one period.
                if self.hb == -1:
                    B_t = tf.constant(0, dtype=tf.float32, shape=self.delta_t_next[:, 0].shape)
                else:
                    impact_bid = tf.cast(tf.less(self.delta_t_next[:, 0], tf.zeros(self.batch_size)), tf.float32) * self.delta_t_next[:, 0] *(-1)
                    B_t = (B_t + impact_bid) * tf.exp(-self.hb * self.h)  # Update impact with
                # transaction and include resilience for one period.

            else:
                self.strategy = tf.concat([self.strategy, tf.expand_dims(self.delta_t_next, axis=0)], axis=0)
                # Compute value in bank account
                diff_delta_t = self.delta_t_next[:, 0] - self.strategy[t - 1, :, 0]
                cashflow = self.liquid_func(self.S_t, -diff_delta_t, A_t, B_t)
                self.Y_t = self.int_rate_bank(self.Y_t) + cashflow  # time-t amount in the bank account
                # Update impact with transaction and include resilience for one period.
                if self.ha == -1:
                    A_t = tf.constant(0, dtype=tf.float32, shape=self.delta_t_next[:, 0].shape)
                else:
                    impact_ask = tf.cast(tf.greater(diff_delta_t, tf.zeros(self.batch_size)), tf.float32) * diff_delta_t
                    A_t = (A_t + impact_ask) * tf.exp(-self.ha * self.h)
                    # Update impact with transaction and include resilience for one period.
                if self.hb == -1:
                    B_t = tf.constant(0, dtype=tf.float32, shape=self.delta_t_next[:, 0].shape)
                else:
                    impact_bid = tf.cast(tf.less(diff_delta_t, tf.zeros(self.batch_size)), tf.float32) * diff_delta_t * (-1)
                    B_t = (B_t + impact_bid) * np.exp(-self.hb * self.h)

            # 5.5) Update features for next time-step (impact and resilience were updated before
            # A) stock price follows GBM
            if (self.stock_dyn == "BSM"):
                #tf.compat.v1.set_random_seed(10) #### Uncomment to get the same random events sequence
                Z = tf.compat.v1.random_normal(shape=[self.batch_size])  # vector of N(0,1)
                self.S_t *= tf.exp((self.mu - self.sigma ** 2 / 2) * self.h + self.sigma * tf.sqrt(self.h) * Z)

            self.S_t_tensor = tf.concat([self.S_t_tensor, tf.expand_dims(self.S_t, axis=0)], axis=0)
            self.delta_t = self.strategy[t, :, 0]  # Number of shares for next period feature vector

            # 5.6) Portfolio value if you were to close
            L_t = self.liquid_func(self.S_t, self.delta_t, A_t, B_t)
            V_t = self.int_rate_bank(self.Y_t) + L_t
            self.V_t_tensor = tf.concat([self.V_t_tensor, tf.expand_dims(V_t, axis=0)], axis=0)
            # Stor trajectories of impact values
            self.A_t_tensor = tf.concat([self.A_t_tensor, tf.expand_dims(A_t, axis=0)], axis=0)
            self.B_t_tensor = tf.concat([self.B_t_tensor, tf.expand_dims(B_t, axis=0)], axis=0)

            # 5.7) Processing of stock price
            self.S_t_pre = self.S_t
            if (self.prepro_stock == "Log-moneyness"):
                self.S_t = tf.math.log(self.S_t_pre / self.strike)
            elif (self.prepro_stock == "Log"):
                self.S_t = tf.math.log(self.S_t_pre)

        # 6) Compute hedging error at maturity - see slide 12/19 of SSC2018 (or paper of C.S.)
        # - check condition for indicator function if worth it to execute or not
        # - currently only working for call option, for put, would need to define the new rules
        self.Y_t = self.int_rate_bank(self.Y_t)
        self.S_t = self.inverse_processing(self.S_t, self.prepro_stock)

        # If call option: buyer execute iif profit selling > K
        if (self.position_type == "Short"):
            if (self.option_type == "Call"):
                self.condition = tf.greater_equal(self.cost_selling(self.S_t, self.nbs_shares, B_t),
                                                  self.nbs_shares * self.strike)
                self.hedging_gain = tf.where(self.condition, self.Y_t + self.liquid_func(self.S_t,
                                                                                         self.delta_t - self.nbs_shares, A_t, B_t) + self.nbs_shares * self.strike,
                                             self.Y_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
                self.hedging_err = -self.hedging_gain
                # If put option: buyer sell iif profit selling < K
            # - i.e. K = 100, S_T = 100.01, maybe will be able to sell only at 99.99 with market impact (execute)
            elif (self.option_type == "Put"):
                self.condition = tf.less_equal(self.cost_selling(self.S_t, self.nbs_shares, B_t),
                                               self.nbs_shares * self.strike)
                self.hedging_gain = tf.where(self.condition, self.Y_t + self.liquid_func(self.S_t,
                                                                                         self.delta_t + self.nbs_shares, A_t, B_t) - self.nbs_shares * self.strike,
                                             self.Y_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
                self.hedging_err = -self.hedging_gain

        elif (self.position_type == "Long"):
            if (self.option_type == "Call"):
                self.condition = tf.greater_equal(self.cost_selling(self.S_t, self.nbs_shares, B_t),
                                                  self.nbs_shares * self.strike)
                self.hedging_gain = tf.where(self.condition, self.Y_t + self.liquid_func(self.S_t,
                                                                                         self.delta_t + self.nbs_shares, A_t, B_t) - self.nbs_shares * self.strike,
                                             self.Y_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
                self.hedging_err = -self.hedging_gain

            elif (self.option_type == "Put"):
                self.condition = tf.less_equal(self.cost_selling(self.S_t, self.nbs_shares, B_t),
                                               self.nbs_shares * self.strike)
                self.hedging_gain = tf.where(self.condition, self.Y_t + self.liquid_func(self.S_t,
                                                                                         self.delta_t - self.nbs_shares, A_t, B_t) + self.nbs_shares * self.strike,
                                             self.Y_t + self.liquid_func(self.S_t, self.delta_t, A_t, B_t))
                self.hedging_err = -self.hedging_gain

                # 7) Compute the loss on the batch of hedging error
        self.loss = self.loss_in_optim()
        
        # 8) Minimize the loss function
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # Computes gradient of the loss with respect to the parameters of the model.
        # Then, applies one-step update rule with the optimizer (i.e. Adam)
        # - Parameters here are strictly the NNet
        self.train = optimizer.minimize(self.loss)

        # 9) Save the model
        self.saver = tf.train.Saver()
        self.model_name = name

    # l_t function computation for bank interest
    # - currently, its periodic, not annualized continuous risk-free rates
    def int_rate_bank(self, x):
        return tf.nn.relu(x) * (1 + self.r_lend) ** self.h - tf.nn.relu(-x) * (1 + self.r_borrow) ** self.h

    # F_t^a function - cost of buying
    # - input: S_t: current spot price
    # -        x  : number of shares to buy
    def cost_buying(self, S_t, x, y):
        return S_t * ((1 + x + y) ** self.alpha - (1 + y) ** self.alpha)

    # F_t^b function - profit from selling
    # - input: S_t: current spot price
    # -        x  : number of shares to sell
    def cost_selling(self, S_t, x, y):
        return S_t * ((1 + x + y) ** self.beta - (1 + y) ** self.beta)

        # function that returns max(0, x) element-wise

    def max_pos(self, x):
        return tf.nn.relu(x)

    # function that returns max(0, -x) element-wise
    def max_neg(self, x):
        return tf.nn.relu(-x)

    # L_t function for liquidation value
    # - 'x': number of shares
    def liquid_func(self, S_t, x, A_t, B_t):
        return self.cost_selling(S_t, tf.nn.relu(x), B_t) - self.cost_buying(S_t, tf.nn.relu(-x), A_t)

    # Given a type of preprocessing, reverse the processing of the stock price
    def inverse_processing(self, paths, prepro_stock):
        if (prepro_stock == "Log-moneyness"):
            paths = tf.multiply(self.strike, tf.exp(paths))  # should broadcast self.strike match
        elif (prepro_stock == "Log"):
            paths = tf.exp(paths)
        return paths

    # Function only works during the optimization, not outside
    def loss_in_optim(self):
        if (self.loss_type == "RMSE"):
            loss = tf.sqrt(tf.reduce_mean(tf.square(self.hedging_err)))
        elif (self.loss_type == "RMSE_per_share"):
            loss = tf.divide(tf.sqrt(tf.reduce_mean(tf.square(self.hedging_err))), self.nbs_shares)
        elif (self.loss_type == "SMSE"):
            loss = tf.reduce_mean(tf.square(tf.nn.relu(self.hedging_err)))
        elif (self.loss_type == "RSMSE"):
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.nn.relu(self.hedging_err))))
        elif (self.loss_type == "RSMSE_per_share"):
            loss = tf.divide(tf.sqrt(tf.reduce_mean(tf.square(tf.nn.relu(self.hedging_err)))), self.nbs_shares)
        return loss

        # Function works outside the optimization - can feed any loss function

    def loss_out_optim(self, hedging_err, loss_type):
        if (loss_type == "RMSE"):
            loss = np.sqrt(np.mean(np.square(hedging_err)))
        elif (loss_type == "RMSE_per_share"):
            loss = np.sqrt(np.mean(np.square(hedging_err))) / self.nbs_shares
        elif (loss_type == "RSMSE"):
            loss = np.sqrt(np.mean(np.square(np.where(hedging_err > 0, hedging_err, 0))))
        elif (loss_type == "RSMSE_per_share"):
            loss = np.sqrt(np.mean(np.square(np.where(hedging_err > 0, hedging_err, 0)))) / self.nbs_shares
        return loss

    # Compute the payoff function depending on the type of assets
    def payoff_func(self, S_T):
        # single asset
        if (self.option_type == "Call"):
            payoff = tf.maximum(S_T - self.strike, 0)
        elif (self.option_type == "Put"):
            payoff = tf.maximum(self.strike - S_T, 0)
        return payoff

    def _execute_graph_batchwise(self, sample_size_train, sess, epochs):
        start = dt.datetime.now()  # compute time
        self.loss_epochs = 9999999 * np.ones(epochs)  # store the loss at the end of each epoch for the train
        loss_best = 999999999
        epoch = 0


        # 0) Loop while we haven't reached the max. epoch and early stopping criteria is not reached
        maxAt = np.array([])
        maxBt = np.array([])

        while (epoch < epochs):
            hedging_err_train = []
            strat = np.array([])
            exercised = np.array([])

            # 1) loop over sample size (train) to do one complete epoch
            nb_rep = int(sample_size_train/self.batch_size)
            for i in range(int(sample_size_train / self.batch_size)):
                # Training step
                _, strat_tmp, hedging_err, input_t_tensor, loss, delta_t_next = sess.run([self.train, self.strategy, self.hedging_err, self.input_t_tensor, self.loss, self.delta_t_next])
                hedging_err_train.append(hedging_err)
                # if i == 0: 
                #     print(hedging_err)
                exercised = np.append(exercised, sess.run(self.condition))
                strat = np.append(strat, np.mean(np.abs(strat_tmp[:,:,0])))
                maxAt = np.append(maxAt, np.max(sess.run(self.A_t_tensor)))
                maxBt = np.append(maxBt, np.max(sess.run(self.B_t_tensor)))

            print("DELTA_T_NEXT: ", delta_t_next)

            # 3) Store the loss on the train after each epoch
            self.loss_epochs[epoch] = self.loss_out_optim(np.concatenate(hedging_err_train), self.loss_type)
            # 4) Print some statistics along the way
            if (epoch + 1) % 1 == 0:
                print('Time elapsed:', dt.datetime.now() - start)
                print('Epoch %d, %s, Train: %.3f' % (epoch + 1, self.loss_type, self.loss_epochs[epoch]))
                print('Proportion of exercise = ', np.mean(exercised))
                print('Strike = ', self.strike)
                # print("Strategy: ", strat)

            # 5) Save the model if it's better
            if (self.loss_epochs[epoch] < loss_best):
                loss_best = self.loss_epochs[epoch]
                self.saver.save(sess, "C:\\Users\\andrei\\Documents\\school\\grad\\y2\\code_tensorflow\\" + self.model_name)
                # We study possible values for the impact

            epoch += 1  # increment the epoch

        # 6) End of the code
       # print("---Finished training results---")
       # print('Time elapsed:', dt.datetime.now() - start)
       # print("Maximum values for the impact")
        print("Max At = ", np.max(maxAt))
        print("Max Bt = ", np.max(maxBt))

        # 7) Return the loss per epoch on the train set
        return self.loss_epochs

    # Out-of-sample function to compute hedging strategy (no training)
    def predict(self, sample_size, sess, epochs):
        batch_size = self.batch_size
        start = dt.datetime.now()  # compute time
        # Save the hedging Pnl for each batch
        hedging_err_pred = []
        strategy_pred = []
        S_t_tensor_pred = []
        V_t_tensor_pred = []
        A_t_tensor_pred = []  #Je copie la strucutre du code, je ne sais pas ce que je fais.
        B_t_tensor_pred = []
        input_t_tensor_pred = []

        # loop over sample size to do one complete epoch
        for i in range(int(sample_size / batch_size)):
            # hedging_err, strategy = sess.run([self.hedging_err, self.strategy])
            # Added tensor versions of A_t and B_t.
            hedging_err, strategy, S_t_tensor, V_t_tensor, A_t_tensor, B_t_tensor, input_t_tensor = sess.run(
                [self.hedging_err, self.strategy, self.S_t_tensor, self.V_t_tensor, self.A_t_tensor, self.B_t_tensor, self.input_t_tensor])

            # This is the batch of input: (nbs_point_traj x indices x 1)
            strategy_pred.append(strategy)
            hedging_err_pred.append(hedging_err)
            S_t_tensor_pred.append(S_t_tensor)
            V_t_tensor_pred.append(V_t_tensor)
            A_t_tensor_pred.append(A_t_tensor)
            B_t_tensor_pred.append(B_t_tensor)
            input_t_tensor_pred.append(input_t_tensor)

        return np.concatenate(strategy_pred, axis=1), np.concatenate(hedging_err_pred), np.concatenate(S_t_tensor_pred, axis=1), np.concatenate(    V_t_tensor_pred, axis=1), np.concatenate(A_t_tensor_pred, axis=1), np.concatenate(B_t_tensor_pred, axis=1), np.concatenate(input_t_tensor_pred, axis=1)
        # return np.concatenate(strategy_pred,axis=1), np.concatenate(hedging_err_pred)

    def restore(self, sess, checkpoint):
        self.saver.restore(sess, checkpoint)

    def point_pred(self, sess, t_t, St, Vt, At, Bt, deltat):
        At = [At]
        Bt = [Bt]
        deltat = [deltat]
        self.deltat = deltat

        if (self.prepro_stock == "Log"):
            self.S_t = tf.math.log(St)
        elif (self.prepro_stock == "Log-moneyness"):
            self.S_t = tf.math.log(St / self.strike)
        elif (self.prepro_stock == "None"):
            self.S_t = St
        self.S_t = [self.S_t]

        #Create the input tensor
        if (self.alpha == 1.00 and self.beta == 1.00):  # no impact, don,t include 'delta_t'
            input_t = tf.concat([tf.expand_dims(self.S_t, axis=1), tf.expand_dims(self.deltat, axis=1),
                                 tf.expand_dims([Vt / self.V_0], axis=1)], axis=1)
        else:
            input_t = tf.concat([tf.expand_dims(self.S_t, axis=1), tf.expand_dims(self.deltat, axis=1),
                                 tf.expand_dims([Vt / self.V_0], axis=1), tf.expand_dims(At, axis=1),
                                 tf.expand_dims(Bt, axis=1)], axis=1)

        self.S_t = self.inverse_processing(self.S_t, self.prepro_stock)

        # 5.4) Compute hedge and update cash amount in bank account
        # Need to add time-step (add h*t, the time-step, a normalization)
        input_t = tf.concat([input_t, tf.expand_dims([self.h * t_t], axis=1)], axis=1)

        layer = self.layer_1(input_t)
        layer = self.layer_2(layer)
        delta_t_next = self.layer_out(layer)


        return sess.run(delta_t_next)[0,0]
