# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 15:58:03 2025

@author: tinghuang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.signal import find_peaks
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


# #%% CCA1/LHY
# file1 = pd.read_csv('CL.csv', header = None, sep = ',') 
# df1 = pd.DataFrame(file1)
# t_data = df1[0]

# y_C_wt_22 = df1[1]
# y_L_wt_22 = df1[2]
# y_C_wt_22_se = df1[3]
# y_L_wt_22_se = df1[4]

# y_C_mu_22 = df1[6]
# y_L_mu_22 = df1[7]
# y_C_mu_22_se = df1[8]
# y_L_mu_22_se = df1[9]


# y_C_wt_29 = df1[11]
# y_L_wt_29 = df1[12]
# y_C_wt_29_se = df1[13]
# y_L_wt_29_se = df1[14]

# y_C_mu_29 = df1[16]
# y_L_mu_29 = df1[17]
# y_C_mu_29_se = df1[18]
# y_L_mu_29_se = df1[19]


# #%% PRR9/PRR7
# file2 = pd.read_csv('P97.csv', header = None, sep = ',') 
# df2 = pd.DataFrame(file2)


# y_P9_wt_22 = df2[1]
# y_P7_wt_22 = df2[2]
# y_P9_wt_22_se = df2[3]
# y_P7_wt_22_se = df2[4]

# y_P9_mu_22 = df2[6]
# y_P7_mu_22 = df2[7]
# y_P9_mu_22_se = df2[8]
# y_P7_mu_22_se = df2[9]


# y_P9_wt_29 = df2[11]
# y_P7_wt_29 = df2[12]
# y_P9_wt_29_se = df2[13]
# y_P7_wt_29_se = df2[14]

# y_P9_mu_29 = df2[16]
# y_P7_mu_29 = df2[17]
# y_P9_mu_29_se = df2[18]
# y_P7_mu_29_se = df2[19]


# #%% PRR5/TOC1
# file3 = pd.read_csv('P51.csv', header = None, sep = ',') 
# df3 = pd.DataFrame(file3)


# y_P5_wt_22 = df3[1]
# y_T_wt_22 = df3[2]
# y_P5_wt_22_se = df3[3]
# y_T_wt_22_se = df3[4]

# y_P5_mu_22 = df3[6]
# y_T_mu_22 = df3[7]
# y_P5_mu_22_se = df3[8]
# y_T_mu_22_se = df3[9]


# y_P5_wt_29 = df3[11]
# y_T_wt_29 = df3[12]
# y_P5_wt_29_se = df3[13]
# y_T_wt_29_se = df3[14]

# y_P5_mu_29 = df3[16]
# y_T_mu_29 = df3[17]
# y_P5_mu_29_se = df3[18]
# y_T_mu_29_se = df3[19]


# #%% ELF4/LUX
# file4 = pd.read_csv('EL.csv', header = None, sep = ',') 
# df4 = pd.DataFrame(file4)


# y_E_wt_22 = df4[1]
# y_LUX_wt_22 = df4[2]
# y_E_wt_22_se = df4[3]
# y_LUX_wt_22_se = df4[4]

# y_E_mu_22 = df4[6]
# y_LUX_mu_22 = df4[7]
# y_E_mu_22_se = df4[8]
# y_LUX_mu_22_se = df4[9]


# y_E_wt_29 = df4[11]
# y_LUX_wt_29 = df4[12]
# y_E_wt_29_se = df4[13]
# y_LUX_wt_29_se = df4[14]

# y_E_mu_29 = df4[16]
# y_LUX_mu_29 = df4[17]
# y_E_mu_29_se = df4[18]
# y_LUX_mu_29_se = df4[19]


file = pd.read_csv('Clockgenes_12L12D.csv', header = None, sep = ',') 
df = pd.DataFrame(file)
t_data = df[0]

CL_data = df[1]
P97_data = df[2]
P51_data = df[3]
EL_data = df[4]
GI_data = df[5]
RL_data = df[6]


CLse_data = df[8]
P97se_data = df[9]
P51se_data = df[10]
ELse_data = df[11]
GIse_data = df[12]
RLse_data = df[13]


#%% light input

def lightdark(t):
    if np.mod(t,24)>=12:
        L = 0
        D = 1
    else:
        L = 1
        D = 0  
    return L,D

#%% Define the tomato circadian model
def tomato_circadian(y,t):
    
    Lavg = 1/2
    Lamp = 1/2
    omega1=2*np.pi/24
    phi1 = 12
    L = Lavg+Lamp*np.cos(omega1*(t-phi1))
    D = 1-L
    
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
    

    
    CLm = y[0]
    CLp = y[1]
   
    P51m= y[2]
    P51p= y[3]
   
    P   = y[4]
    
    
    z=np.array([(v1+v1L*L*P)/(1+(CLp/K0)**2+(P51p/K2)**2)-(k1L*L+k1D*D)*CLm,\
                (p1+p1L*L)*CLm-d1*CLp,\
                v3/(1+(CLp/K6)**2+(P51p/K7)**2)-k3*P51m,\
                p3*P51m-(d3D*D+d3L*L)*P51p,\
                0.3*(1-P)*D-L*P])
    

    return z

def tomato_circadian1(y,t):
    
    L,D = lightdark(t)
    
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
    

    
    CLm = y[0]
    CLp = y[1]
   
    P51m= y[2]
    P51p= y[3]
   
    P   = y[4]
    
    
    z=np.array([(v1+v1L*L*P)/(1+(CLp/K0)**2+(P51p/K2)**2)-(k1L*L+k1D*D)*CLm,\
                (p1+p1L*L)*CLm-d1*CLp,\
                v3/(1+(CLp/K6)**2+(P51p/K7)**2)-k3*P51m,\
                p3*P51m-(d3D*D+d3L*L)*P51p,\
                0.3*(1-P)*D-L*P])
    

    return z


tspan=np.array([0,120]);
h=0.01
t=np.arange(tspan[0],tspan[1]+h,h)
y0 = [0.73504743, 1.13705128, 0.2264693,  0.38493123,  0.69340948]

yy1 = odeint(tomato_circadian, y0, t)
yy2 = odeint(tomato_circadian1, y0, t)

font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 24,}

font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 24,}

x = np.linspace(12,24,2)
y1 = 0*np.ones(len(x))
y2 = 3*np.ones(len(x))
for i in range(10):
    x1 = x+24*i
    plt.fill_between(x1,y1,y2,facecolor='silver')
    
plt.plot(t-72,(yy1[:,0]-min(yy1[720:,0]))/(max(yy1[720:,0])-min(yy1[720:,0])),'k-',label='Continuous',linewidth=3)
plt.plot(t-72,(yy2[:,0]-min(yy2[720:,0]))/(max(yy2[720:,0])-min(yy2[720:,0])),'b-',label='Discrete',linewidth=3)

plt.xlabel('Time(h)',font)
plt.ylabel('Relative $\it{CL}$ mRNA level',font1)

plt.scatter(t_data,CL_data*1.1,marker='o',color='red',label='$\it{SlCCA1}$(Exp.)')
plt.errorbar(t_data,CL_data*1.1,CLse_data,lw = 0,ecolor='red',elinewidth=2,ms=7,capsize=3)
plt.plot(t_data,CL_data*1.1,linestyle = '--',color='r',linewidth=3)

plt.title('module4',font)

plt.xticks(fontproperties = 'Times New Roman', size = 24)
plt.yticks(fontproperties = 'Times New Roman', size = 24)
plt.xlim([0,48])
plt.ylim([0,3])

plt.legend(prop={'family' : 'Times New Roman', 'size'   : 20})


x_major_locator=MultipleLocator(6)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


# plt.subplot(222)
# x = np.linspace(12,24,2)
# y1 = 0*np.ones(len(x))
# y2 = 3*np.ones(len(x))
# for i in range(10):
#     x1 = x+24*i
#     plt.fill_between(x1,y1,y2,facecolor='silver')
    
# plt.plot(t-72,yy1[:,2],'k-',label='Continuous',linewidth=3)
# plt.plot(t-72,yy2[:,2],'b-',label='Discrete',linewidth=3)

# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative $\it{P97}$ mRNA level',font1)


# plt.scatter(t_data,P97_data,marker='o',color='red',label='$\it{SlPRR9}$(Exp.)')
# plt.errorbar(t_data,P97_data,P97se_data,lw = 0,ecolor='red',elinewidth=2,ms=7,capsize=3)
# plt.plot(t_data,P97_data,linestyle = '--',color='r',linewidth=3)


# plt.xticks(fontproperties = 'Times New Roman', size = 24)
# plt.yticks(fontproperties = 'Times New Roman', size = 24)
# plt.xlim([0,48])
# plt.ylim([0,3])
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 20})


# x_major_locator=MultipleLocator(6)
# y_major_locator=MultipleLocator(0.5)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)

# plt.subplot(223)
# x = np.linspace(12,24,2)
# y1 = 0*np.ones(len(x))
# y2 = 3*np.ones(len(x))
# for i in range(10):
#     x1 = x+24*i
#     plt.fill_between(x1,y1,y2,facecolor='silver')
    
# plt.plot(t-72,yy1[:,4],'k-',label='Continuous',linewidth=3)
# plt.plot(t-72,yy2[:,4],'b-',label='Discrete',linewidth=3)


# plt.scatter(t_data,P51_data*1.1,marker='o',color='red',label='$\it{SlTOC1}$(Exp.)')
# plt.errorbar(t_data,P51_data*1.1,P51se_data,lw = 0,ecolor='red',elinewidth=2,ms=7,capsize=3)
# plt.plot(t_data,P51_data*1.1,linestyle = '--',color='r',linewidth=3)

# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative $\it{P51}$ mRNA level',font1)


# plt.xticks(fontproperties = 'Times New Roman', size = 24)
# plt.yticks(fontproperties = 'Times New Roman', size = 24)
# plt.xlim([0,48])
# plt.ylim([0,3])
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 20})


# x_major_locator=MultipleLocator(6)
# y_major_locator=MultipleLocator(0.5)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)


# plt.subplot(224)
# x = np.linspace(12,24,2)
# y1 = 0*np.ones(len(x))
# y2 = 3*np.ones(len(x))
# for i in range(10):
#     x1 = x+24*i
#     plt.fill_between(x1,y1,y2,facecolor='silver')
    
# plt.plot(t-72,yy1[:,6],'k-',label='Continuous',linewidth=3)
# plt.plot(t-72,yy2[:,6],'b-',label='Discrete',linewidth=3)

# plt.scatter(t_data,EL_data*1.2,marker='o',color='red',label='$\it{SlELF4}$(Exp.)')
# plt.errorbar(t_data,EL_data*1.2,ELse_data,lw = 0,ecolor='red',elinewidth=2,ms=7,capsize=3)
# plt.plot(t_data,EL_data*1.2,linestyle = '--',color='r',linewidth=3)

# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative $\it{EL}$ mRNA level',font1)


# plt.xticks(fontproperties = 'Times New Roman', size = 24)
# plt.yticks(fontproperties = 'Times New Roman', size = 24)
# plt.xlim([0,48])
# plt.ylim([0,3])
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 20})

# x_major_locator=MultipleLocator(6)
# y_major_locator=MultipleLocator(0.5)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)





