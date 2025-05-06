# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:49:16 2025

@author: Ting Huang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.signal import find_peaks
from matplotlib.pyplot import MultipleLocator

# Define the tomato circadian model
def tomato_circadian(t, y, k3):
    CLm, CLp, P97m, P97p, P51m, P51p, ELm, ELp, P, u, v = y
    v1=1.58
    v1L=3.0
    v2A=1.27
    v2L=5.0
    v3=1.0
    v4=0.02
    v4L=1.47
    k1L=0.53
    k1D=0.35
    k2=0.75
    # k3=0.56
    k4=0.37   
    p1=0.76
    p1L=0.42
    p2=1.01
    p3=0.64
    p4=1.01
    d1=0.68
    d2D=0.5 
    d2L=0.29 
    d3D=0.48
    d3L=0.38
    d4D=1.21  
    d4L=0.38  
    K0=2.8
    K1=0.16
    K2=1.18
    K3=1.73
    K4=0.28
    K5=0.57
    K6=0.46
    K7=2.0
    K8=0.36
    K9=1.9
    K10=1.9
    
    a1 = 0.1
    a2 = 0.685
    b1 =-0.05
    b2 = 0.343
    
    L = u
    D = 1-L
    
    dCLm_dt = (v1+v1L*L*P)/(1+(CLp/K0)**2+(P97p/K1)**2+(P51p/K2)**2)-(k1L*L+k1D*D)*CLm
    dCLp_dt = (p1+p1L*L)*CLm-d1*CLp
    dP97m_dt = (v2L*L*P+v2A)/(1+(CLp/K3)**2+(P51p/K4)**2+(ELp/K5)**2)-k2*P97m
    dP97p_dt = p2*P97m-(d2D*D+d2L*L)*P97p
    dP51m_dt = v3/(1+(CLp/K6)**2+(P51p/K7)**2)-k3*P51m
    dP51p_dt = p3*P51m-(d3D*D+d3L*L)*P51p
    dELm_dt = (v4+L*v4L)/(1+(CLp/K8)**2+(P51p/K9)**2+(ELp/K10)**2)-k4*ELm
    dELp_dt = p4*ELm-(d4D*D+d4L*L)*ELp
    dP_dt = 0.3*(1-P)*D-L*P
    du_dt = v*a1+b1
    dv_dt = -u*a2+b2
    
    return [dCLm_dt, dCLp_dt, dP97m_dt, dP97p_dt, dP51m_dt, dP51p_dt, dELm_dt, dELp_dt, dP_dt, du_dt, dv_dt]

# Time span for integration
t_span = (0, 96)
t_eval = np.linspace(t_span[0], t_span[1], 960)


# tspan=np.array([0,48]);
# h=0.01
# t=np.arange(tspan[0],tspan[1]+h,h)

# Initial conditions
y0 = [0.85371592e+00, 0.72626332e+00, 5.01968466e-01, 6.46276320e-01,2.25236529e-01, 5.94439074e-01, 1.27049114e-02, 2.01664004e-02,0.5,0.9,0.1]

# Parameter range for bifurcation analysis
k_deg_values = np.linspace(0, 0.2, 300)
initial_conditions1 = np.linspace(0.8, 1.2, 50)

periods = np.zeros((len(initial_conditions1), len(k_deg_values)))
phases = np.zeros((len(initial_conditions1), len(k_deg_values)))

# Perform bifurcation analysis
for i, y0_factor in enumerate(initial_conditions1):
    y0_scaled = [y * y0_factor for y in y0]
    for j, k_deg in enumerate(k_deg_values):
        # sol = odeint(tomato_circadian,  y0_scaled, t, (k_deg,))
        sol = solve_ivp(tomato_circadian, t_span, y0_scaled, args=(k_deg,), t_eval=t_eval)
        
        # Find oscillation peaks
        p2_values = sol.y[0]
        peaks, _ = find_peaks(p2_values[480:])
        # peaks = np.where((p2_values[:-1] < p2_values[1:]) )[0]
        
        if len(peaks) > 1:
            period = np.mean(np.diff(sol.t[peaks]))
            phase = sol.t[peaks[1]] % period  # Phase relative to first peak
            periods[i, j] = period
            phases[i, j] = phase-12
        
        else:
            periods[i, j] = np.nan
            phases[i, j] = np.nan

# Plot heatmap diagrams
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 22,}

font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 20,}

plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlim([0,0.2])
plt.ylim([0.8,1.2])

c1 = ax[0].imshow(periods, aspect='auto', origin='lower', extent=[k_deg_values.min(), k_deg_values.max(), initial_conditions1.min(), initial_conditions1.max()], cmap='viridis')
ax[0].set_xlabel('$k_{3}$', fontdict=font)
ax[0].set_ylabel('Period (h)', fontdict=font)
# ax[0].set_title('Period Heatmap')

# ax[0].set_title('Period Heatmap')
cb1 = fig.colorbar(c1, ax=ax[0])
cb1.ax.tick_params(labelsize=20)
cb1.set_label("", fontdict={'family': 'Times New Roman', 'size': 20})
plt.rcParams['font.family'] = 'Times New Roman'


plt.xlim([0,0.2])
plt.ylim([0.8,1.2])

c2 = ax[1].imshow(phases, aspect='auto', origin='lower', extent=[k_deg_values.min(), k_deg_values.max(), initial_conditions1.min(), initial_conditions1.max()], cmap='plasma')
ax[1].set_xlabel('$k_{3}$', fontdict=font)
ax[1].set_ylabel('Phase (h)', fontdict=font)
# ax[1].set_title('Phase Heatmap')
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
# ax[1].set_title('Phase Heatmap')
cb2 = fig.colorbar(c2, ax=ax[1])
cb2.ax.tick_params(labelsize=20)
cb2.set_label("", fontdict={'family': 'Times New Roman', 'size': 20})
plt.rcParams['font.family'] = 'Times New Roman'

x_major_locator=MultipleLocator(0.05)
y_major_locator=MultipleLocator(0.1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.tight_layout()
plt.show()