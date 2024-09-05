#Hedging error for discrete delta hedging in Black-Scholes model using Monte Carlo simulation.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

#parametres du modèle
r=0.0
mu=0.1
sigma = 0.1898
T=10 #jours
nban=252 #nb jours par an
nbper=10 #nb periodes
dt=1/nban #discretisation (annuelle)
nbsim=10000 #nb simulations
K=1000 #prix d'exercice
S0=1000
b=1100 # barriere

param = [r,mu,sigma,T,nbper,dt,nbsim,K,S0,b]

def trajectoire(x,param):
	S = [0 for i in range(param[4]+1)]
	S[0]=x
	
	r=param[0]
	sigma=param[2]
	nbper=param[4]
	dt=param[5]
	
	for i in range(nbper):
		S[i+1] = S[i]+ S[i]*r*dt + S[i]*sigma*np.sqrt(dt)*np.random.normal()
		
	return S

def payoff_Call(x,param):
	K = param[7]
	return (x - K) * (x > K)
	
#t is the number of periods since t=0.
def delta_BS(St, t, param):
	r = param[0]
	sigma = param[2]
	dt = param[5]
	K = param[7]
	T = param[3] * dt
	t = t * dt
	delta_t = T-t
	
	d1 = 1/(sigma * np.sqrt(delta_t)) * (np.log(St/K) + (r + sigma * sigma/2) * delta_t)
	return norm.cdf(d1)
	
def BS_price(param):
	r = param[0]
	sigma = param[2]
	S0 = param[8]
	dt = param[5]
	K = param[7]
	delta_t = param[3] * dt
		
	d1 = 1/(sigma * np.sqrt(delta_t)) * (np.log(S0/K) + (r + sigma * sigma/2) * delta_t)
	d2 = d1 - sigma * np.sqrt(delta_t)
	
	return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r*delta_t)
		
	
def hedging(param, St):
	r = param[0]
	sigma = param[2]
	S0 = param[8]
	dt = param[5]
	K = param[7]
	nbper = param[4]
	V0 = BS_price(param)
	Vt = [0 for i in range(nbper +1)]
	Vt[0] = [BS_price(param)]
	for i in range(len(St)-1):
		del1 = delta_BS(St[i], i, param)
		del0 = (Vt[i] - del1 * St[i]) * np.exp( - r * i * dt)
		Vt[i+1] = Vt[i] + del1* (St[i+1] - St[i]) 
	return Vt[-1]

def hedging_err(param, nbsim):
	err = [0 for i in range(nbsim)]
	S0 = param[8]
	
	for i in range(nbsim):
		St = trajectoire(S0,param)
		VT = hedging(param, St)
		err[i] = VT - payoff_Call(St[-1], param)
	
	return np.mean(err), np.std(err)
	

