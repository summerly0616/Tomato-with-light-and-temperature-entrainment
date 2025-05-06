# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:49:16 2025

@author: Ting Huang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from matplotlib.pyplot import MultipleLocator

# Define the tomato circadian model
def tomato_circadian(t, y, d3D):
    CLm, CLp, P97m, P97p, P51m, P51p, ELm, ELp, P, u, v, s, w = y
    # v1=1.58
    # v1L=3.0
    # v2A=1.27
    # v2L=5.0
    # v3=1.0
    # v4=0.02
    # v4L=1.47
    # k1L=0.53
    # k1D=0.35
    # k2=0.75
    # k3=0.56
    # k4=0.37   
    # p1=0.76
    # p1L=0.42
    # p2=1.01
    # p3=0.64
    # p4=1.01
    # # d1=0.68
    # d2D=0.5 
    # d2L=0.29 
    # d3D=0.48
    # d3L=0.38
    # d4D=1.21  
    # d4L=0.38  
    # K0=2.8
    # K1=0.16
    # K2=1.18
    # K3=1.73
    # K4=0.28
    # K5=0.57
    # K6=0.46
    # K7=2.0
    # K8=0.36
    # K9=1.9
    # K10=1.9
    
    a1 = 0.1
    a2 = 0.685
    b1 =-0.05
    b2 = 0.343
    
    L = u
    D = 1-L
    
    a3 = 0.16
    a4 = 0.428
    b3 = -4
    b4 = 10.709
    
    T = s+273.5
    R = 8.3145*1e-3
    
    T1=T
    T2=T
    
    E11=50
    E12=50
    
    
    Par1_A = 1.58*np.exp(E11/R/T1)
    Par2_A = 3.0*np.exp(E11/R/T1)
    Par3_A = 1.27*np.exp(E11/R/T1)
    Par4_A = 0.02*np.exp(E11/R/T1)
    Par5_A = 5.0*np.exp(E11/R/T1)
    Par6_A = 1.0*np.exp(E11/R/T1)
    Par7_A = 1.47*np.exp(E11/R/T1)
    Par8_A = 0.53*np.exp(E11/R/T1)
    Par9_A = 0.35*np.exp(E11/R/T1)
    Par10_A = 0.75*np.exp(E11/R/T1)
    Par11_A = 0.56*np.exp(E11/R/T1)
    Par12_A = 0.37*np.exp(E11/R/T1)
    Par13_A = 0.76*np.exp(E11/R/T1)
    Par14_A = 0.42*np.exp(E11/R/T1)
    Par15_A = 1.01*np.exp(E11/R/T1)
    Par16_A = 0.64*np.exp(E11/R/T1)
    Par17_A = 1.01*np.exp(E11/R/T1)
    Par18_A = 0.68*np.exp(E11/R/T1)
    Par19_A = 0.5*np.exp(E11/R/T1)
    Par20_A = 0.29*np.exp(E11/R/T1)
    Par21_A = 0.48*np.exp(E11/R/T1)
    Par22_A = 0.38*np.exp(E11/R/T1)
    Par23_A = 1.21*np.exp(E11/R/T1)
    Par24_A = 0.38*np.exp(E11/R/T1)
    Par25_A = 0.2*np.exp(E11/R/T1)
    Par26_A = 1.2*np.exp(E11/R/T1)
    Par27_A = 0.2*np.exp(E11/R/T1)
    Par28_A = 0.2*np.exp(E11/R/T1)
    Par29_A = 0.3*np.exp(E11/R/T1)
    Par30_A = 0.5*np.exp(E11/R/T1)
    Par31_A = 2.0*np.exp(E11/R/T1)
    Par32_A = 0.4*np.exp(E11/R/T1)
    Par33_A = 1.9*np.exp(E11/R/T1)
    Par34_A = 1.9*np.exp(E11/R/T1)
    Par35_A = 2.8*np.exp(E11/R/T1)
    
    PPar1_A = 1.58*np.exp(E12/R/T2)
    PPar2_A = 3.0*np.exp(E12/R/T2)
    PPar3_A = 1.27*np.exp(E12/R/T2)
    PPar4_A = 0.02*np.exp(E12/R/T2)
    PPar5_A = 5.0*np.exp(E12/R/T2)
    PPar6_A = 1.0*np.exp(E12/R/T2)
    PPar7_A = 1.47*np.exp(E12/R/T2)
    PPar8_A = 0.53*np.exp(E12/R/T2)
    PPar9_A = 0.35*np.exp(E12/R/T2)
    PPar10_A = 0.75*np.exp(E12/R/T2)
    PPar11_A = 0.56*np.exp(E12/R/T2)
    PPar12_A = 0.37*np.exp(E12/R/T2)
    PPar13_A = 0.76*np.exp(E12/R/T2)
    PPar14_A = 0.42*np.exp(E12/R/T2)
    PPar15_A = 1.01*np.exp(E12/R/T2)
    PPar16_A = 0.64*np.exp(E12/R/T2)
    PPar17_A = 1.01*np.exp(E12/R/T2)
    PPar18_A = 0.68*np.exp(E12/R/T2)
    PPar19_A = 0.5*np.exp(E12/R/T2)
    PPar20_A = 0.29*np.exp(E12/R/T2)
    PPar21_A = 0.48*np.exp(E12/R/T2)
    PPar22_A = 0.38*np.exp(E12/R/T2)
    PPar23_A = 1.21*np.exp(E12/R/T2)
    PPar24_A = 0.38*np.exp(E12/R/T2)
    PPar25_A = 0.2*np.exp(E12/R/T2)
    PPar26_A = 1.2*np.exp(E12/R/T2)
    PPar27_A = 0.2*np.exp(E12/R/T2)
    PPar28_A = 0.2*np.exp(E12/R/T2)
    PPar29_A = 0.3*np.exp(E12/R/T2)
    PPar30_A = 0.5*np.exp(E12/R/T2)
    PPar31_A = 2.0*np.exp(E12/R/T2)
    PPar32_A = 0.4*np.exp(E12/R/T2)
    PPar33_A = 1.9*np.exp(E12/R/T2)
    PPar34_A = 1.9*np.exp(E12/R/T2)
    PPar35_A = 2.8*np.exp(E12/R/T2)
    
    
    s1 = 0.5 * (1 + np.sin(2 * np.pi * (t - 8) / 24))
    s2 = 1-s1
    
    
    v1  = (PPar1_A*s2+Par1_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    v1L = (PPar2_A*s2+Par2_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    v2A = (PPar3_A*s2+Par3_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    v4 = (PPar4_A*s2+Par4_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    v2L = (PPar5_A*s2+Par5_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    v3  = (PPar6_A*s2+Par6_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    v4L  = (PPar7_A*s2+Par7_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    k1L = (PPar8_A*s2+Par8_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    k1D = (PPar9_A*s2+Par9_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    k2  = (PPar10_A*s2+Par10_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    k3  = (PPar11_A*s2+Par11_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    k4  = (PPar12_A*s2+Par12_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    p1  = (PPar13_A*s2+Par13_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    p1L = (PPar14_A*s2+Par14_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    p2  = (PPar15_A*s2+Par15_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    p3  = (PPar16_A*s2+Par16_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    p4  = (PPar17_A*s2+Par17_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    d1  = (PPar18_A*s2+Par18_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    d2D = (PPar19_A*s2+Par19_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    d2L = (PPar20_A*s2+Par20_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    # d3D = (PPar21_A*s2+Par21_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    d3L = (PPar22_A*s2+Par22_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    d4D = (PPar23_A*s2+Par23_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    d4L = (PPar24_A*s2+Par24_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K1  = (PPar25_A*s2+Par25_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K2  = (PPar26_A*s2+Par26_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K3  = (PPar27_A*s2+Par27_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K4  = (PPar28_A*s2+Par28_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K5  = (PPar29_A*s2+Par29_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K6  = (PPar30_A*s2+Par30_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K7  = (PPar31_A*s2+Par31_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K8  = (PPar32_A*s2+Par32_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K9  = (PPar33_A*s2+Par33_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K10 = (PPar34_A*s2+Par34_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    K0 = (PPar35_A*s2+Par35_A*s1)*np.exp(-(E12*s2+E11*s1)/(R*T))
    
    
    
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
    ds_dt = w*a3+b3
    dw_dt = -s*a4+b4
    
    return [dCLm_dt, dCLp_dt, dP97m_dt, dP97p_dt, dP51m_dt, dP51p_dt, dELm_dt, dELp_dt, dP_dt, du_dt, dv_dt, ds_dt, dw_dt]

# Time span for integration
t_span = (0, 96)
t_eval = np.linspace(t_span[0], t_span[1], 960)

# Initial conditions
y0 = [0.85371592e+00, 0.72626332e+00, 5.01968466e-01, 6.46276320e-01,2.25236529e-01, 5.94439074e-01, 1.27049114e-02, 2.01664004e-02,0.5,0.9,0.1,16,14]

# Parameter range for bifurcation analysis
k_deg_values = np.linspace(0, 0.2, 100)
initial_conditions1 = np.linspace(0.8, 1.2, 50)
initial_conditions2 = np.linspace(0, 6, 50)
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

font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 22,}

font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 20,}

# Plot heatmap diagrams
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlim([0,0.2])
plt.ylim([0.8,1.2])

c1 = ax[0].imshow(periods, aspect='auto', origin='lower', extent=[k_deg_values.min(), k_deg_values.max(), initial_conditions1.min(), initial_conditions1.max()], cmap='viridis')
ax[0].set_xlabel('$d_{3D}$', fontdict=font)
ax[0].set_ylabel('Period (h)', fontdict=font)
# ax[0].set_title('Period Heatmap')
cb1 = fig.colorbar(c1, ax=ax[0])
cb1.ax.tick_params(labelsize=20)
cb1.set_label("", fontdict={'family': 'Times New Roman', 'size': 20})
plt.rcParams['font.family'] = 'Times New Roman'


plt.xlim([0,0.2])
plt.ylim([0.8,1.2])

c2 = ax[1].imshow(phases, aspect='auto', origin='lower', extent=[k_deg_values.min(), k_deg_values.max(), initial_conditions1.min(), initial_conditions1.max()], cmap='plasma')
ax[1].set_xlabel('$d_{3D}$', fontdict=font)
ax[1].set_ylabel('Phase (h)', fontdict=font)
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