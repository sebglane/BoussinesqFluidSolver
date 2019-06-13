#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:36:10 2019

@author: sg
"""
import numpy as np
import os
from os import path
results_directory = "/media/Volume/2D_rotating_run"
results = np.loadtxt(path.join(results_directory, "benchmark_report.txt"),
                     skiprows=1)
time = results[:,1]
phi = results[:,2]
temperature = results[:,3]
v_phi = results[:,4]

omega = np.zeros(phi.shape)
omega[0:-1] = np.diff(phi) / np.diff(time)
omega[-1] = (phi[-1] - phi[-2]) / (time[-1] - time[-2])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=4, sharex=True)
ax[0].plot(time, phi)
ax[0].plot(time[139], phi[139], 'x')
ax[1].plot(time, temperature)
ax[2].plot(time, v_phi)
ax[3].plot(time, omega)

