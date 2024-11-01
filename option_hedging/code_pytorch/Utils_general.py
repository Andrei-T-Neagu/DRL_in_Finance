import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import poisson
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import norm
import math

def n_timesteps_func(T, rebalancement_timeframe):
    if (rebalancement_timeframe == "daily"):
        n_timesteps = int(260 * T)  # nbs of timesteps (rebalancement), and not timestep
    elif (rebalancement_timeframe == "weekly"):
        n_timesteps = int(260 * T / 5)
    elif (rebalancement_timeframe == "quarterly"):
        n_timesteps = int(4 * T)
    elif (rebalancement_timeframe == "monthly"):
        print("Watch out, different computations (360 days assumed)")
        n_timesteps = int(360 * T / 30)
    elif (rebalancement_timeframe == "yearly"):
        n_timesteps = int(T)
    return n_timesteps


# -------------------------------------------------------------------------------------------------#
# -----------------------Preprocessing/inverse of the stock price----------------------------------#
# -------------------------------------------------------------------------------------------------#
# Validate: CHECK
def preprocessing(paths, prepro_stock, strike, barrier=-99999999):
    if (prepro_stock == "Log-moneyness"):
        paths = np.log(paths / strike)
    elif (prepro_stock == "Log-barrier"):
        paths = np.log(paths / barrier)
    elif (prepro_stock == "Log"):
        paths = np.log(paths)
    elif (prepro_stock == "Nothing"):
        return paths
    return paths


# Single set of paths that you need to use the bijection inversion
# Validate: CHECK
def inverse_processing(paths, prepro_stock, strike, barrier):
    if (prepro_stock == "Log-moneyness"):
        paths = strike * np.exp(paths)
    elif (prepro_stock == "Log-money-barrier"):
        paths = barrier * strike * np.exp(paths)
    elif (prepro_stock == "Log-barrier"):
        paths = barrier * np.exp(paths)
    elif (prepro_stock == "Log"):
        paths = np.exp(paths)
    return paths


# -------------------------------------------------------------------------------------------------#
# Small check if the rebalancement frequence is the same as the payoff frequence computation      #
# - ex: monthly rebal, but yearly observation for payoff computation in VA's
# -------------------------------------------------------------------------------------------------#
def freq_obs_freq_rebal_func(hedging_instruments, rebalancement_timeframe):
    if (hedging_instruments == 'Stock' and rebalancement_timeframe == 'monthly'):
        freq_obs_equal_freq_rebal = False  # rebalancement is monthly, but payoff freq is yearly
    else:
        freq_obs_equal_freq_rebal = True
    return freq_obs_equal_freq_rebal


def print_stats(hedging_err, deltas, loss_type, model_name, V_0, riskaversion=99999999999,
                print_loss=False, output=True):
    # add dimension to deltas when necessary,
    if len(deltas.shape) == 2:
        deltas = np.expand_dims(deltas, axis=2)
    # 1) All statistics are computed w.r. to the hedging error
    # A) Everything except for the average turnover
    mean_hedging_err, CVaR_95, CVaR_99, VaR_95, VaR_99, MSE, semi_MSE, skew_, kurt_, max_, min_, std = hedging_stats(
        hedging_err, deltas)

    # B) Turnover computation
    turnover = turnover_func(deltas)

    # 2) Print out everything
    # if (output):
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.hist(hedging_err, bins=50)
    #     plt.xlabel('Hedging error')
    #     plt.ylabel('Frequency')
    #     plt.title('%s' % (model_name))
    #     plt.savefig('%s.png' % (model_name))

    if (loss_type == "CVaR" or loss_type == "CVaR_per_share"):
        loss = np.mean(np.sort(hedging_err)[int(riskaversion * hedging_err.shape[0]):])

    if (print_loss):
        if (loss_type == "CVaR" or loss_type == "CVaR_per_share"):
            print("----------------------------------------------------------")
            print('CVaR: %.3f, with %.3f level of risk.' % (loss, riskaversion))
            print("----------------------------------------------------------")

        elif (loss_type == "Neg_exp"):
            print("----------------------------------------------------------")
            print('Neg Expo loss: %.3f, with %.3f risk aversion parameter.' % (loss, riskaversion))
            print("----------------------------------------------------------")

            # Show additional results
    print('Model was trained with the loss function: %s' % (loss_type))
    print('Initial investment:', V_0)
    print('Mean Hedging error:', mean_hedging_err)
    print('CVaR_95: %.4f, CVaR_99: %.4f' % (CVaR_95, CVaR_99))
    print('VaR_95: %.4f, VaR_99: %.4f' % (VaR_95, VaR_99))
    print('MSE: %.4f, RMSE: %.4f' % (MSE, np.sqrt(MSE)))
    print('Semi-MSE: %.4f, Semi-RMSE: %.4f' % (semi_MSE, np.sqrt(semi_MSE)))
    print('Skew: %.4f, Kurt: %.4f' % (skew_, kurt_))
    print('Max error: %.4f, Min error: %.4f' % (max_, min_))
    print('STD error: %.4f' % (std))
    print('Avg turnover of underlying: %.4f' % (turnover))
    RMSE = np.sqrt(MSE)
    return (hedging_err, deltas, RMSE, std )


# -------------------------------------------------------------------------------------------------#
# S_T          : normalize stock prices
# strike       : real value of the strike price
# option_type  : {"call", "put"}
# -------------------------------------------------------------------------------------------------#
def payoff_liability_func(underlying_unorm_prices, strike, option_type, freq_obs_equal_freq_rebal, barrier):
    # single asset
    S_T = underlying_unorm_prices[-1, :]
    if (option_type == "call"):
        payoff = np.maximum(S_T - strike, 0)
    elif (option_type == "put"):
        payoff = np.maximum(strike - S_T, 0)
    elif (option_type == "Lookback_fixed_strike_put"):
        payoff = np.maximum(strike - np.amin(underlying_unorm_prices, axis=0), 0)
    elif (option_type == "Asian_avg_price_put"):
        Z_T = np.mean(underlying_unorm_prices, axis=0)
        payoff = np.maximum(strike - Z_T, 0)
    elif (option_type == "Barrier_up_and_out_put"):
        # true if barrier crossed, false if never crossed
        condition = np.greater_equal(np.amax(underlying_unorm_prices, axis=0),
                                     barrier)  # Barrier need to be applied element-wise
        # if true ---> Z_T = 0.0, if false ----> 1.0
        Z_T = np.where(condition, 0.0, 1.0)
        payoff = Z_T * np.maximum(strike - S_T, 0)  # Z_T should be applied element-wise

    # Embedded option in variable annuity for the rachet GBDM
    # - payoff = max(H_T-S_T,0) where H_T = max {S_0,...,S_T-1}
    elif (option_type == "Lookback_put_VA"):
        if (freq_obs_equal_freq_rebal):
            payoff = np.maximum(np.amax(underlying_unorm_prices[:-1, :], axis=0) - underlying_unorm_prices[-1, :], 0)
        else:
            idx = np.array(np.arange(0, int(10 * 12), 12))
            payoff = np.maximum(np.amax(underlying_unorm_prices[idx, :], axis=0) - underlying_unorm_prices[-1, :], 0)
    return payoff


# ------------------------------------------------------------------------------------#
# hedging_err: losses for each paths (i.e. negative is a gain, positive is a loss)
# ------------------------------------------------------------------------------------#
def hedging_stats(hedging_err, deltas):
    mean_hedging_err = np.mean(hedging_err)
    CVaR_95 = np.mean(np.sort(hedging_err)[int(0.95 * hedging_err.shape[0]):])
    CVaR_99 = np.mean(np.sort(hedging_err)[int(0.99 * hedging_err.shape[0]):])
    VaR_95 = np.sort(hedging_err)[int(0.95 * hedging_err.shape[0])]
    VaR_99 = np.sort(hedging_err)[int(0.99 * hedging_err.shape[0])]
    MSE = np.mean(np.square(hedging_err))
    semi_MSE = np.mean(np.square(np.where(hedging_err > 0, hedging_err, 0)))
    skew_ = skew(hedging_err)
    kurt_ = kurtosis(hedging_err)
    max_ = np.max(hedging_err)
    min_ = np.min(hedging_err)
    std = np.std(hedging_err)
    return mean_hedging_err, CVaR_95, CVaR_99, VaR_95, VaR_99, MSE, semi_MSE, skew_, kurt_, max_, min_, std


def turnover_func(deltas):
    # The mean is taken with respect to the number of time-steps (to compare across different maturities)
    turnover = np.mean(np.sum(np.absolute(deltas[1:, :, :] - deltas[0:-1, :, :]), axis=0), axis=0)
    return (turnover)


def print_turnover(turnover, hedging_instruments, nbs_assets):
    print('Avg turnover of %s: %.4f' % (hedging_instruments, turnover))


# ----------------------------------------------------------------------------------------------#
#                            Black-scholes Prices Functions                                    #
# ----------------------------------------------------------------------------------------------#
# VALIDATE: CHECK
def BS_d1(S, dt, r, sigma, strike):
    return (np.log(S / strike) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))


# VALIDATE: CHECK
# Style : +1 for call, -1 for put
def BlackScholes_price(S, T, r, sigma, strike, style, t=0):
    dt = T - t
    Phi = stats.norm(loc=0, scale=1).cdf
    d1 = BS_d1(S, dt, r, sigma, strike)
    d2 = d1 - sigma * np.sqrt(dt)
    return style * S * Phi(style * d1) - style * strike * np.exp(-r * dt) * Phi(style * d2)


# VALIDATE: CHECK
# Style : +1 for call, -1 for put
def BS_delta(S, T, r, sigma, strike, style, t=0):
    dt = T - t
    d1 = BS_d1(S, dt, r, sigma, strike)
    Phi = stats.norm(loc=0, scale=1).cdf
    delta_call = Phi(d1)
    if (style == 1):
        result = delta_call
    else:
        result = delta_call - 1
    return result


# L_t function for liquidation value
def liquid_func(S_t, x, trans_costs):
    return cost_selling(S_t, np.maximum(x, 0), trans_costs) - cost_buying(S_t, np.maximum(-x, 0), trans_costs)


# l_t function computation for bank interest
# - currently, its periodic, not annualized continuous risk-free rates
def int_rate_bank(x, r_lend, r_borrow, h):
    return np.maximum(x, 0) * (1 + r_lend) ** h - np.maximum(-x, 0) * (1 + r_borrow) ** h


# F_t^a function - cost of buying
# - input: S_t: current spot price
# -        x  : number of shares to buy
def cost_buying(S_t, x, trans_costs):
    return S_t * x + trans_costs * x


# F_t^b function - profit from selling
# - input: S_t: current spot price
# -        x  : number of shares to sell
def cost_selling(S_t, x, trans_costs):
    return S_t * x - trans_costs * x

# Function to compute delta-hedging using given trajectories of the geometric BM with resilience
# set hab = [-1,-1] for no impact case
def delta_hedge_res(St_traj, r_borrow, r_lend, sigma, T, option_type, position_type, strike, V_0, nbs_shares, trans_costs = 0.0, Leland=False):

    time_vect_len, nb_traj = St_traj.shape  # nb of time points (counting t=0), nb of trajectories
    N = time_vect_len - 1 # step-size
    delta_t = T / N  # Maturity in years/nb of time intervals

    if Leland:
        sigma = sigma*math.sqrt(1+math.sqrt(2/math.pi)*(trans_costs/(sigma*math.sqrt(delta_t))))

    V_t = np.zeros(St_traj.shape)
    A_t = np.zeros(St_traj.shape)
    B_t = np.zeros(St_traj.shape)

    deltas = np.zeros([N, nb_traj])

    if (position_type == "long"):
        V_t[0, :] = -V_0 * np.ones(nb_traj)

    # - if short, you receive the premium that you put in the bank
    elif (position_type == "short"):
        V_t[0, :] = V_0 * np.ones(nb_traj)

    for t in range(N):
        if option_type == 'call':
            deltas[t, :] = nbs_shares * BS_delta(St_traj[t, :], T - t * delta_t, (r_borrow + r_lend) / 2, sigma, strike, 1)

        elif option_type == 'put':
            deltas[t, :] = nbs_shares * BS_delta(St_traj[t, :], T - t * delta_t, (r_borrow + r_lend) / 2, sigma, strike, -1)

        if (t == 0):
            cashflow = liquid_func(St_traj[t, :], -deltas[t, :], trans_costs)  # cashflow of first investment done
            Y_t = V_t[t, :] + cashflow  # time-0 amount in the bank account

        else:
            diff_delta_t = deltas[t, :] - deltas[t - 1, :]
            cashflow = liquid_func(St_traj[t, :], -diff_delta_t, trans_costs)
            Y_t = int_rate_bank(Y_t, r_lend, r_borrow, delta_t) + cashflow  # time-t amount in the bank account

        L_t = liquid_func(St_traj[t+1, :], deltas[t, :], trans_costs)
        V_t = Y_t + L_t

    if (position_type == "short"):
        if (option_type == "call"):
            condition = np.greater_equal(cost_selling(St_traj[-1, :], nbs_shares, trans_costs), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(St_traj[-1, :], deltas[-1, :] - nbs_shares, trans_costs) + nbs_shares * strike,
                                    Y_t + liquid_func(St_traj[-1, :], deltas[-1, :], trans_costs))

        elif (option_type == "put"):
            condition = np.less_equal(cost_selling(St_traj[-1, :], nbs_shares, trans_costs), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(St_traj[-1, :], deltas[-1, :] + nbs_shares, trans_costs) - nbs_shares * strike,
                                    Y_t + liquid_func(St_traj[-1, :], deltas[-1, :], trans_costs))

    elif (position_type == "long"):
        if (option_type == "call"):
            condition = np.greater_equal(cost_selling(St_traj[-1, :], nbs_shares, trans_costs), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(St_traj[-1, :], deltas[-1, :] + nbs_shares, trans_costs) - nbs_shares * strike,
                                    Y_t + liquid_func(St_traj[-1, :], deltas[-1, :], trans_costs))

        elif (option_type == "put"):
            condition = np.less_equal(cost_selling(St_traj[-1, :], nbs_shares, trans_costs), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(St_traj[-1, :], deltas[-1, :] - nbs_shares, trans_costs) + nbs_shares * strike,
                                    Y_t + liquid_func(St_traj[-1, :], deltas[-1, :], trans_costs))

    hedging_err = -hedging_gain

    return deltas, hedging_err

# Function to compute delta-hedging results
# - currently working with:
#    - Dynamics: BSM only
#    - Option  : call only
def delta_hedge(nbs_point_traj, test_paths, r_borrow, r_lend, mu, sigma, S_0, T, alpha, beta,
                option_type, position_type, strike, V_0, nbs_shares):
    # - if long, you buy the option by borrowing V_0 from bank
    if (position_type == "long"):
        V_t = -V_0

    # - if short, you receive the premium that you put in the bank
    elif (position_type == "short"):
        V_t = V_0

    N = nbs_point_traj - 1
    h = T / N  # step-size
    S_t = S_0  # time-0 stock price
    deltas = np.zeros((N, test_paths, 1))

    for t in range(N):  # loop over time-step
        # compute time-t hedging strategy
        if (option_type == "call"):
            deltas[t, :, 0] = nbs_shares * BS_delta(S_t, T - t * h, (r_borrow + r_lend) / 2, sigma, strike, 1)
        elif (option_type == "put"):
            deltas[t, :, 0] = nbs_shares * BS_delta(S_t, T - t * h, (r_borrow + r_lend) / 2, sigma, strike, -1)

        # A) Update bank account value
        if (t == 0):
            cashflow = liquid_func(S_t, -deltas[t, :, 0], alpha, beta)  # cashflow of first investment done
            Y_t = V_t + cashflow  # time-0 amount in the bank account

        else:
            diff_delta_t = deltas[t, :, 0] - deltas[t - 1, :, 0]
            cashflow = liquid_func(S_t, -diff_delta_t, alpha, beta)
            Y_t = int_rate_bank(Y_t, r_lend, r_borrow, h) + cashflow  # time-t amount in the bank account

        # B) stock price follow GBM
        Z = np.random.randn(test_paths)  # vector of N(0,1)
        S_t *= np.exp((mu - sigma ** 2 / 2) * h + sigma * np.sqrt(h) * Z)

        # D) Portfolio value if you were to close
        # L_t          = cost_selling(S_t, np.maximum(deltas[t,:,0],0), beta) - cost_buying(S_t, np.maximum(-deltas[t,:,0],0), alpha)
        L_t = liquid_func(S_t, deltas[t, :, 0], alpha, beta)
        V_t = Y_t + L_t

        # Compute hedging error
    if (position_type == "short"):
        if (option_type == "call"):
            condition = np.greater_equal(cost_selling(S_t, nbs_shares, beta), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(S_t, deltas[-1, :, 0] - nbs_shares, alpha,
                                                                 beta) + nbs_shares * strike,
                                    Y_t + liquid_func(S_t, deltas[-1, :, 0], alpha, beta))

        elif (option_type == "put"):
            condition = np.less_equal(cost_selling(S_t, nbs_shares, beta), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(S_t, deltas[-1, :, 0] + nbs_shares, alpha,
                                                                 beta) - nbs_shares * strike,
                                    Y_t + liquid_func(S_t, deltas[-1, :, 0], alpha, beta))

    elif (position_type == "long"):
        if (option_type == "call"):
            condition = np.greater_equal(cost_selling(S_t, nbs_shares, beta), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(S_t, deltas[-1, :, 0] + nbs_shares, alpha,
                                                                 beta) - nbs_shares * strike,
                                    Y_t + liquid_func(S_t, deltas[-1, :, 0], alpha, beta))

        elif (option_type == "put"):
            condition = np.less_equal(cost_selling(S_t, nbs_shares, beta), nbs_shares * strike)
            hedging_gain = np.where(condition, Y_t + liquid_func(S_t, deltas[-1, :, 0] - nbs_shares, alpha,
                                                                 beta) + nbs_shares * strike,
                                    Y_t + liquid_func(S_t, deltas[-1, :, 0], alpha, beta))

    hedging_err = -hedging_gain

    return deltas, hedging_err


def MertonCDF(x, param, h):
    # Idea:
    # - pdf of jump-diffusion gaussian is known in closed-form since X(t)|N(t)=k follows a gaussian distribution;
    #   - it's an infinite gaussian mixture;

    x_flatten = x.flatten()  # becomes a vector of size[M,], not a matrix
    [mu, sigma, lambda_, gamma, delta] = param

    # Rescale the parameters
    mu1 = mu * h
    sig1 = sigma * np.sqrt(h)
    jumpIntensity = lambda_ * h
    kappa = np.exp(gamma + delta * delta / 2) - 1  # kappa = E[exp(Xi_j)] - 1where Xi_{j} is N(gamma, delta^2).
    a = mu1 - jumpIntensity * kappa - 0.5 * sig1 * sig1  # drift term in the model

    # 1) Gets the 1-1E^10 percentile of poisson process
    # - These will be used for the proportional term in the gaussian mixture
    K = int(poisson.ppf(1 - 1E-10, jumpIntensity))  # it works, it's '57'
    tt = np.arange(0, int(poisson.ppf(1 - 1E-10, jumpIntensity)) + 1)  # [0,1.,,,,K]

    # 2) Compute parameter terms of the gaussian part in the mixture
    prob = poisson.pmf(tt,
                       jumpIntensity)  # [p(0), p(1),...,p(K)] where p(j) is the probability mass function of poisson
    mu2 = a + tt * gamma  # conditional mean of jump-diffusion, i.e. at + k*gamma if N(t) = k.
    sig2 = np.sqrt(sig1 ** 2 + tt * delta ** 2)  # conditional std of jump-duffison
    m = len(x_flatten)
    xmat = np.tile(np.array([x_flatten]).transpose(), (1, (K + 1)))  # np.tile is the equivalent of repmat in Matlab
    mumat = np.tile(mu2, (m, 1))
    sigmat = np.tile(sig2, (m, 1))

    # Compute for each value of jump [0,1,...,K] the conditional CDF for each point
    # 1) Compute the CDF of the gaussian term for each jump              --- > gives 'P'
    # 2) Multiply by the probability of observing 'j' jumps, i.e. (prob) --- > gives 'F'
    Z = (xmat - mumat) / sigmat
    Phi = stats.norm(loc=0, scale=1).cdf
    P = Phi(Z)
    F = np.sum(P * np.expand_dims(prob, axis=1).transpose(), axis=1)
    return np.reshape(F, (x.shape))


# ----------------------------------------------------------------------------------------------#
# To price call and put option as well as the deltas
# - This is exact computation with closed-form of CDF
# - Parameters given as input are risk-neutral
def MertonOption(s, K, r, tau, sigma, lambda_, gamma, delta):
    # why (5) parameters? params[0,1] = 0, it will be the drift term of the excess return under Q.
    params = np.zeros((5, 2))
    params[1, 1] = sigma
    params[2, 1] = lambda_
    params[3, 1] = gamma
    params[4, 1] = delta
    params[:, 0] = params[:, 1]

    # kappa(s) = E[exp(s*zeta_k)], see slide 23/31
    kappa_1 = np.exp(gamma + 0.5 * delta ** 2) - 1;  # kappa(1)
    kappa_2 = np.exp(2 * gamma + 0.5 * 2 * 2 * delta ** 2) - 1;  # kappa(2)

    # These are the parameters for Pi1:
    # - See slide 23-24/32 of 5bOptionsFourier.pdf;
    # - lambda^{e} = lambda*(1+kappa(1));
    # - gamma^{e}  = gamma + delta^2
    # - kappa^e(s) = exp(s*gamma^e + s^2*delta^2/2)-1;
    # - sigma^e    = sigma  (no change)
    # - delta^e    = delta  (no change)
    # -------------------------------------------------------------#
    params[0, 0] = sigma ** 2 + lambda_ * (kappa_2 - 2 * kappa_1)
    params[2, 0] = lambda_ * (1 + kappa_1);  # This is lambda^{e}
    params[3, 0] = gamma + delta ** 2;  # This is gamma^{e}

    x = -np.log(s / K) - r * tau;  # 'x' can be a matrix of pairs of paths and points on the path

    # evaluate at 'x' the CDF. Recall that Pi1 and Pi2 are SURVIVAL functions, not CDF, hence the 1-CDF
    # - see slide 20/31, eq. (6) and (7), where the CDF form is used, not the characteristic function in this case

    # Also important:
    # - This is the computation of CDF of excess return, i.e. log(S_T/S_t) - rTau, which has a 'zero' drift term for 'alpha'.
    #   - Which is why params[0,1] = 0!
    Pi2 = 1 - MertonCDF(x, params[:, 1], tau)  # for pi_2, use parameters under Q
    Pi1 = 1 - MertonCDF(x, params[:, 0], tau)  # for pi_1, use parameters under Q^{e} (the Esscher transform)

    # s*exp(x) = s*exp(-log(s/K)-r*tau) = s*K/s*exp(-rtau) = K*exp(-rtau), so it's perfectly fine!
    call_value = s * (Pi1 - np.exp(x) * Pi2);
    call_delta = Pi1;
    put_value = call_value - s * (1 - np.exp(x));
    put_delta = Pi1 - 1;

    return call_value, call_delta, put_value, put_delta


# ---------------------------------------------------------- #
# Code to compute hedging err of a given hedging strategy
def test_hedging_strategy_new(nbs_point_traj, deltas, paths, strike, T, r_borrow, r_lend, alpha, beta,
                              option_type, position_type, V_0, nbs_shares):
    # - if long, you buy the option by borrowing V_0 from bank
    if (position_type == "long"):
        V_t = -V_0

    # - if short, you receive the premium that you put in the bank
    elif (position_type == "short"):
        V_t = V_0

    N = nbs_point_traj - 1
    h = T / N  # step-size
    print(h)

    for t in range(N):  # loop over time-step
        # compute time-t hedging strategy

        # A) Update bank account value
        if (t == 0):
            cashflow = liquid_func(paths[t, :], -deltas[t, :, 0], alpha, beta)  # cashflow of first investment done
            Y_t = V_t + cashflow  # time-0 amount in the bank account

        else:
            diff_delta_t = deltas[t, :, 0] - deltas[t - 1, :, 0]
            cashflow = liquid_func(paths[t, :], -diff_delta_t, alpha, beta)
            Y_t = int_rate_bank(Y_t, r_lend, r_borrow, h) + cashflow  # time-t amount in the bank account

        # B) Portfolio value if you were to close
        L_t = liquid_func(paths[t, :], deltas[t, :, 0], alpha, beta)
        V_t = Y_t + L_t

        # Compute hedging error
    condition = np.greater_equal(cost_selling(paths[-1, :], nbs_shares, beta), nbs_shares * strike)
    hedging_gain = np.where(condition, Y_t + liquid_func(paths[-1, :], deltas[-1, :, 0] - nbs_shares, alpha,
                                                         beta) + nbs_shares * strike,
                            Y_t + liquid_func(paths[-1, :], deltas[-1, :, 0], alpha, beta))
    hedging_err = -hedging_gain

    return hedging_err


# -------------------------------------------------------------------------------------------------#
# paths        : unormalized stock price
# strike       : real value of the strike price
# option_type  : {"call", "put"}
# -------------------------------------------------------------------------------------------------#
def payoff_func(paths, strike, option_type, barrier=-999999999):
    # single asset
    if (option_type == "call"):
        S_T = paths[-1, :]
        payoff = np.maximum(S_T - strike, 0)
    elif (option_type == "put"):
        S_T = paths[-1, :]
        payoff = np.maximum(strike - S_T, 0)
    elif (option_type == "Lookback_fixed_strike_put"):
        payoff = np.maximum(strike - np.amin(paths, axis=0), 0)

    elif (option_type == "Asian_avg_price_put"):
        Z_T = np.mean(paths, axis=0)
        payoff = np.maximum(strike - Z_T, 0)
    elif (option_type == "Barrier_up_and_out_put"):
        # true if barrier crossed, false if never crossed
        condition = np.greater_equal(np.amax(paths, axis=0), barrier)  # Barrier need to be applied element-wise
        # if true ---> Z_T = 0.0, if false ----> 1.0
        Z_T = np.where(condition, 0.0, 1.0)
        S_T = paths[-1, :]
        payoff = Z_T * np.maximum(strike - S_T, 0)  # Z_T should be applied element-wise
    return payoff


# Function to plot hedge across spot prices of different trading strategies
# input:
# - spot_prices : vector of S_t prices (x-axis of the plot)
# - delta_hedge : vector of hedges corresponding to the spot prices
# - global_hedge: vector of global hedge corresponding to the spot prices
def plot_hedge(spot_prices, delta_hedge, global_hedge, title_name, save_name):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(spot_prices, delta_hedge, label="Delta hedge")
    ax.plot(spot_prices, global_hedge, label="Global hedge")
    plt.xlabel('Spot price', fontsize=16)
    plt.ylabel('Share of stock', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("%s" % (title_name), fontsize=20)
    plt.legend(fontsize=20)
    ax.grid()
    plt.show()
    fig.savefig(save_name)