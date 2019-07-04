#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:36:10 2019

@author: sg
"""
import numpy as np
import os
from os import path

max_ind = -1

files = []

results_directory = "/media/Volume/2D_nonrotating_run/"
files.append(path.join(results_directory, "explicit_dt_5e-3",  "global_avg_report.txt"))
files.append(path.join(results_directory, "explicit_dt_2.5e-3",  "global_avg_report.txt"))
files.append(path.join(results_directory, "explicit_dt_1.25e-3",  "global_avg_report.txt"))
labels = ["2.5e-3", "5e-3", "1e-2", "1e-1"]
marker = ["o-" , "-", "+-", "-"]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=3, sharex=True)

for f, l in zip(files, labels):
    results = np.loadtxt(f, skiprows=1)
    time = results[:max_ind,1]
    velocity_rms = results[:max_ind,2]
    kinetic_energy = results[:max_ind,3]
    temperature_rms = results[:max_ind,4]
    
    ax[0].plot(time, velocity_rms, label=l)
    ax[1].plot(time, kinetic_energy, label=l)
    ax[2].plot(time, temperature_rms, label=l)
    
ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel("velocity")
ax[1].set_ylabel("kinetic energy")
ax[2].set_ylabel("temperature")
ax[2].set_xlabel("time")

plt.legend()