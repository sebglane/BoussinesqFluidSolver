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
results = np.loadtxt(path.join(results_directory, "global_avg_report.txt"),
                     skiprows=1)
time = results[:,1]
velocity_rms = results[:,2]
kinetic_energy = results[:,3]
temperature_rms = results[:,4]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(time, velocity_rms)
ax[1].plot(time, kinetic_energy)
ax[2].plot(time, temperature_rms)