# -*- coding: utf-8 -*-
"""
Function to generate a Gaussian with given mean mu and variance var.
Inspired by https://towardsdatascience.com/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968
"""

import numpy as np

def gauss(mu, var, x):
    gaussian_distribution = 1/np.sqrt(2.0*np.pi*var)*np.exp(-(x-mu)**2/(2*var))
    return gaussian_distribution