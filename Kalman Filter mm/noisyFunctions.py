"""
Functions for generating noisy signals
"""
import numpy as np
import matplotlib.pyplot as plt
from random import *
from pandas import Series



def noisePlotter(): #plots a noisy signal alongside the noise-free signal
    N = 1000 #number of time steps
    x = np.zeros(N) #noisy signal to be plotted
    signal = np.zeros(N) #pure, noise-free signal

    for i in range(N):
        signal[i] = np.sin(i/20) #signal to be plotted as a function of i
        noise = uniform(-signal/3, signal/3)    
        x[i] = signal[i] + noise[i]

    t = np.arange(0, N);
    fig, axs = plt.subplots(2)
    fig.suptitle("Noisy signal vs pure signal")
    axs[0].plot(t, x)
    axs[1].plot(t, signal)

    plt.show()



def addNoiseToSignal(input_signal:Series, mean, sdev): #adds noise whith given mean and standard deviation to input signal 
    seed(1)
    N = len(input_signal)
    output_signal = np.zeros(N)
    for i in range(N):
        noise = gauss(mean, sdev)
        output_signal[i] = input_signal[i] + noise
        
    return output_signal
