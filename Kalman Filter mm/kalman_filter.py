# -*- coding: utf-8 -*-
"""
Generate N noisy random measurements with mean mu and variance var
Made by Ben Holm with inspiration from "Fundamentals of Sensor Fusion" by Edmund Brekke
"""
import numpy as np
import matplotlib.pyplot as plt
import noisyFunctions as nf

#Initialization
dim = 1 #Dimension of state vector


F = np.identity(dim) #Transition matrix
H = np.array([1]) #Measurement matrix
Q = np.array([0.01]) #Model uncertainty matrix
R = np.array([0.1]) #Measurement uncertainty matrix

#Prediction step
def predict(x_prior, P_prior, z_k):
    x_pred = F @ x_prior #Predicted state estimate vector
    P_pred = F @ P_prior @ (F.T) + Q #Predicted covariance matrix
    z_pred = H @ x_pred #Predicted measurement vector
    return x_pred, P_pred, z_pred
    
#Innovation (intermediary step)
def innovation(z_pred, z_k, P_pred):
    nu = z_k - z_pred #Innovation 
    if P_pred.size == 1:
        S_k = H @ P_pred * (H.T) + R #Innovation covariance matrix
    else:
        S_k = H @ P_pred @ (H.T) + R
    return nu, S_k

#Update step
def update(nu, S_k, P_pred, x_pred):
    if S_k.size == 1:
        W_k = P_pred * H.T * (1/S_k) #Kalman gain
        #print("x_pred = ", x_pred)
        x_update = x_pred + W_k * nu #Posterior state estimate
        P_update = (1 - W_k * H) * P_pred #Posterior covariance matrix
    else:
        W_k = P_pred @ H.T @ np.linalg.inv(S_k) #Kalman gain
        x_update = x_pred + W_k @ nu #Posterior state estimate
        WkH = W_k @ H #Intermediate processing
        dim = WkH.ndim #Dimension of W_k @ H
        P_update = (np.identity(dim) - W_k @ H) @ P_pred #Posterior covariance matrix
    
    return x_update, P_update

def kalman(x_prior, P_prior, z_k):
    x_pred, P_pred, z_pred = predict(x_prior, P_prior, z_k) #Prediction step
    
    nu, S_k = innovation(z_pred, z_k, P_pred) #Innovation
    
    x_update, P_update = update(nu, S_k, P_pred, x_pred) #Update step
    
    return x_update, P_update, P_pred, S_k


N = 100 #No. of measurements
z_true = [1 for i in range(N)] #True signal vector
z_mea = nf.addNoiseToSignal(z_true, 0, 0.1) #Measurement vector

x_prior, P_prior = np.array([0.2]), np.array([0.5]) #Initial guesses for x and P
x_est = [0 for i in range(N)] #Initialize estimation vector
x_est[0] = x_prior

for k in range(1, N): #Kalman filter over the measurements
    x_update, P_update, P_pred, S_k = kalman(x_prior, P_prior, z_mea[k])
    x_est[k] = x_update 
    x_prior, P_prior = x_update, P_update



#Plots
fig, axs = plt.subplots(2)
axs[0].plot(z_true, color='r', label='True signal')
axs[0].legend(loc='upper right')
axs[1].plot(z_mea, color='b', label='Measurements')
axs[1].plot(x_est, color='magenta', label="Estimate")
axs[1].legend(loc='upper right')
plt.show()



