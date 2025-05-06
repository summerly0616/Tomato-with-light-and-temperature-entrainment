# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:49:16 2025

@author: Ting Huang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.pyplot import MultipleLocator

# Define the tomato circadian model
def tomato_circadian(t, y, d3D):
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
    k3=0.56
    k4=0.37   
    p1=0.76
    p1L=0.42
    p2=1.01
    p3=0.64
    p4=1.01
    d1=0.68
    d2D=0.5 
    d2L=0.29 
    # d3D=0.48
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

# Parameter sweep for bifurcation analysis
k1L_values = np.linspace(0.01, 0.2, 1000)  # Control parameter
stable_states = []
font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 22,}

for k1L in k1L_values:
    sol = solve_ivp(tomato_circadian, [0, 1200], [0.85371592e+00, 0.72626332e+00, 5.01968466e-01, 6.46276320e-01,2.25236529e-01, 5.94439074e-01, 1.27049114e-02, 2.01664004e-02,0.5,0.9,0.1], args=(k1L,), t_eval=np.linspace(0, 1200, 1000))
    stable_states.append(sol.y[:, -1])  # Store steady-state values

stable_states = np.array(stable_states)

# Plot bifurcation diagram
plt.figure(figsize=(8, 6))
plt.plot(k1L_values, stable_states[:, 0], 'bo', markersize=3, label='CL mRNA')
plt.plot(k1L_values, stable_states[:, 4], 'ro', markersize=3, label='P51 mRNA')
plt.xlabel('$d_{3D}$',font)
plt.ylabel('Steady states',font)
plt.legend()
# plt.title('Bifurcation Diagram')
plt.xlim([0,0.2])
plt.ylim([0,2])

x_major_locator=MultipleLocator(0.05)
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 20})
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.show()