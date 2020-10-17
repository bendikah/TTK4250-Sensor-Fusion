# -*- coding: utf-8 -*-
"""
Example of generate_gaussian.py
"""
import matplotlib.pyplot as plt
import generate_gaussian as gg
import numpy as np

mu = 0
var = 2
x_axis = np.arange(-10, 10, 0.1)
g = [gg.gauss(mu, var, x) for x in x_axis]
plt.plot(x_axis, g)