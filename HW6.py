# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:20:30 2023

@author: jomil
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("C")
dpdz =np.asfarray(data.iloc[:,0])
period = 1/3
omega = 2*np.pi/period
time = np.linspace(0,period,len(dpdz))

plt.scatter(time,dpdz)
plt.title("dP/dz vs. Time [Original signal]")
plt.ylabel("Pressure Gradient [mmHg/cm]")
plt.xlabel("Time [s]")


phi_naught = (1/period)*np.trapz(dpdz,dx=np.diff(time)[0])
print("Phi naught = {:.3f}".format(phi_naught))

n = np.linspace(1,10,10).astype(int)
phi_n = []
psi_n = []
for i in n:
    phi = (2/period)*np.trapz(dpdz*np.cos(i*omega*time),dx=np.diff(time)[0])
    phi_n.append(phi)
    psi = (2/period)*np.trapz(dpdz*np.sin(i*omega*time),dx=np.diff(time)[0])
    psi_n.append(psi)
phi_n = np.asfarray(phi_n)
psi_n = np.asfarray(psi_n)
print("Phi n =")
for i in n:
    print("{}th Harmonic: {:.3}".format(i,phi_n[i-1]))
print()
print("Psi n =")
for i in n:
    print("{}th Harmonic: {:.3}".format(i,psi_n[i-1]))

fourier_dpdz = np.zeros(len(time))
for i in n:
    inter = (phi_n[i-1]*np.cos(i*omega*time) + psi_n[i-1]*np.sin(i*omega*time))
    fourier_dpdz += inter
fourier_dpdz += phi_naught
plt.figure()
plt.scatter(time,fourier_dpdz,c="r")
plt.title("dP/dz vs. time [Fourier representation N = {}]".format(max(n)))
plt.ylabel("Pressure Gradient [mmHg/cm]")
plt.xlabel("Time [s]")
