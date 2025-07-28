# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:14:48 2025

@author: Administrator
"""
import numpy as np;
import matplotlib.pyplot as plt
from scipy.linalg import logm

def GetCmu(W,State,i):
    W=np.transpose(W);
    W=np.conjugate(W);
    Eigval,Eigvec=np.linalg.eig(W);
    Eigvec=np.transpose(Eigvec);
    ei=Eigvec;
    cup=0;
    n=len(Eigval);
    for ii in range(n):
        for j in range(n-ii-1):
            if np.real(Eigval[j])<np.real(Eigval[j+1]):
                cup=Eigval[j+1];
                Eigval[j+1]=Eigval[j];
                Eigval[j]=cup;
                cupoftea=Eigvec[j+1].copy();
                Eigvec[j+1]=Eigvec[j];
                Eigvec[j]=cupoftea.copy();
    Cmu=0;
    for ii in range(n):
        Cmu=Cmu+np.conjugate(Eigvec[i][ii])*State[ii];
    return Cmu
def TimeEvo(W,State1,State2,timescale):
    I = np.eye(7, dtype=complex)
    zero_block = np.zeros((7, 7), dtype=complex)
    Projection = np.block([I, zero_block])
    N=timescale*1000;
    eigenvalues,eigenvectors=np.linalg.eig(W);

    for i in range(len(eigenvalues)):
        if np.linalg.norm(eigenvalues[i]) < 1e-10:
            MaxE = i;
            break;
    VecA1 = np.linalg.inv(eigenvectors) @ State1;
    VecA2 = np.linalg.inv(eigenvectors) @ State2;
    eigenvectors=np.transpose(eigenvectors);
    fs=eigenvectors[MaxE];
    fs1= Projection @ fs;
    list1=[];
    list2=[];
    timelist=[];
    rho1=np.zeros((3,3),dtype=complex);
    rho2=np.zeros((3,3),dtype=complex);
    for i in range(N):
        State1n=np.zeros(14);
        State2n=np.zeros(14);
        timelist.append(timescale*i/N)
        for j in range(len(eigenvalues)):
            if j == MaxE:
                State1n = State1n + fs;
                State2n = State2n + fs;
            else:
                State1n = State1n + VecA1[j]*np.exp(eigenvalues[j]*timescale*i/N)*eigenvectors[j];
                State2n = State2n + VecA2[j]*np.exp(eigenvalues[j]*timescale*i/N)*eigenvectors[j];
        State1n = Projection @ State1n;
        State2n = Projection @ State2n;
        """ density matrix """
        for j in range(3):
            rho1[j][j]=State1n[j];
            rho2[j][j]=State2n[j];
        rho1[0][1]=State1n[3];rho2[0][1]=State2n[3];
        rho1[0][2]=State1n[4];rho2[0][2]=State2n[4];
        rho1[1][2]=State1n[5];rho2[1][2]=State2n[5];
        rho1[1][0]=np.conj(rho1[0][1]);
        rho1[2][0]=np.conj(rho1[0][2]);
        rho1[2][1]=np.conj(rho1[2][1]);
        rho2[1][0]=np.conj(rho2[0][1]);
        rho2[2][0]=np.conj(rho2[0][2]);
        rho2[2][1]=np.conj(rho2[2][1]);
        """ density matrix """

        list1.append(np.log10(np.linalg.norm(State1n-fs1)));
        list2.append(np.log10(np.linalg.norm(State2n-fs1)));
    return timelist,list1,list2;
def check(list1,list2):
    j=0;
    n=len(list1);
    for i in range(n):
        if abs(list1[i]-list2[i])<0.03:
            j=i;
            break;
    return j;
np.set_printoptions(precision=3)
r_21 =1;
r_32 = 1;
r_31 = 8;
v = 0.1;
Delta1 = 0.4;
w_2 = 1;
w_3 = 2;
N_21=0;N_31=0;N_32=0;
g1=-0.5*r_21*(1+2*N_21)-0.5*r_31*N_31-0.5*r_32*N_32+1j*w_2;
g2=-0.5*r_31*(1+2*N_31)-0.5*r_32*(1+N_32)-0.5*r_21*N_21+1j*w_3;
g3=-0.5*r_21*(1+N_21)-0.5*r_31*(1+N_31)-0.5*r_32*(1+2*N_32)-1j*(w_2-w_3);
g4=-0.5*r_21*(1+N_21)-0.5*r_31*(1+N_31)-0.5*r_32*(1+2*N_32)-1j*(w_3-w_2);
W0 = np.zeros((7, 7), dtype=complex)
I = np.eye(7, dtype=complex)
zero_block = np.zeros((7, 7), dtype=complex)
Projection = np.block([I, zero_block])
# Projection operator
Delta =  np.zeros((7, 7), dtype=complex);
""" define W0 """
W0[0][0]=-r_31*N_31-r_21*N_21;W0[0][1]=r_21*(1+N_21);W0[0][2]=r_31*(1+N_31)
W0[1][0]=r_21*N_21; W0[1][1]=-r_21*(1+N_21)-r_32*N_32;W0[1][2]=r_32*(1+N_32);
W0[2][0]=r_31*N_31;W0[2][1]=r_32*N_32;W0[2][2]=-r_32*(1+N_32)-r_31*(1+N_31);
W0[3][3]=g1;W0[4][4]=g2;W0[5][5]=g3;W0[6][6]=g4;
""" define W0 """

"""define  \Delta """
Delta[1][5]=1;Delta[1][6]=-1;
Delta[2][5]=-1;Delta[2][6]=1;
Delta[3][4]=Delta[4][3]=Delta[5][1]=Delta[6][2]=1;
Delta[6][1]=Delta[5][2]=-1;
"""define \Delta """
W = np.block([
    [W0,         1j*Delta1*Delta],
    [1j*Delta1*Delta, W0 - v * I]
])

""" define initial state """
zero_vec = np.zeros(7);
stat1=np.array([0.3,0.7,0,0,0,0,0]);
stat2=np.array([0.75,0,0.25,0,np.sqrt(3)/4,0,0]);
State1=np.concatenate([stat1, zero_vec])
State2=np.concatenate([stat2, zero_vec])
""" define initial state """

#time evolution setting
timescale=14;
timelist,list1,list2=TimeEvo(W, State1, State2, timescale);
Deltalist=[];
Cmulist1=[]
Cmulist2=[];
Delta1=0.4
W = np.block([
    [W0,         1j*Delta1*Delta],
    [1j*Delta1*Delta, W0 - v * I]
])
print('D(0) for state 1 = \n',pow(10,list1[0]));
print('D(0) for state 2 = \n',pow(10,list2[0]));
print('a3 for state 1=\n',np.abs(GetCmu(W, State1, 4)))
print('a3 for state 2=\n',np.abs(GetCmu(W, State2, 4)))
print('Cmu for state 1=\n',np.abs(GetCmu(W, State1, 3)))
print('Cmu for state 2=\n',np.abs(GetCmu(W, State2, 3)))

for i in range(100):
    Delta1=0.4/100*i;
    W = np.block([
        [W0,         Delta * 1j * Delta1],
        [Delta * 1j * Delta1, W0 - v * I]
    ])
    Deltalist.append(Delta1);
    Cmulist1.append(np.abs(GetCmu(W,State1,3)))
    Cmulist2.append(np.abs(GetCmu(W, State2, 3)))
  

W = np.block([
    [W0,         0*Delta],
    [0*Delta, W0 - v * I]
])
timelist,list1d,list2d=TimeEvo(W, State1, State2, timescale)

stat1=np.array([0.8,0.1,0.1,0,0,0,0]);
stat2=np.array([8/9,0,1/9,0,np.sqrt(2)*2/4,0,0]);
State1=np.concatenate([stat1, zero_vec])
State2=np.concatenate([stat2, zero_vec])
timelist,list3,list4=TimeEvo(W, State1, State2, timescale);
Delta1=0.4
"""define \Delta """
W = np.block([
    [W0,         Delta * 1j * Delta1],
    [Delta * 1j * Delta1, W0 - v * I]
])
print('D(0) for state 3 = \n',pow(10,list3[0]));
print('D(0) for state 4 = \n',pow(10,list4[0]));
print('Cmu for state 3=\n',np.abs(GetCmu(W, State1, 3)))
print('Cmu for state 4=\n',np.abs(GetCmu(W, State2, 3)))
print('a3 for state 3=\n',np.abs(GetCmu(W, State1, 4)))
print('a3 for state 4=\n',np.abs(GetCmu(W, State2, 4)))
Cmulist3=[]
Cmulist4=[];
timelist,list3d,list4d=TimeEvo(W, State1, State2, timescale)
for i in range(100):
    Delta1=0.4/100*i;
    W = np.block([
        [W0,         Delta * 1j * Delta1],
        [Delta * 1j * Delta1, W0 - v * I]
    ])
    
    Cmulist3.append(np.abs(GetCmu(W,State1,3)))
    Cmulist4.append(np.abs(GetCmu(W, State2, 3)))
fig, axs = plt.subplots(2, 2, figsize=(10, 6),dpi=300)  # 1行2列，图形尺寸10x5英寸
plt.rc('xtick', labelsize=14)  # x轴刻度字体大小
plt.rc('ytick', labelsize=15)

# 在左侧子图绘制数据
axs[0,0].plot(timelist, list1d, color='blue',linestyle = '-',label=r'$p_\alpha$')
axs[0,0].plot(timelist,list2d, color = 'green',linestyle ='-',label=r'$p_\beta$')
axs[0,0].set_xlabel(r'$t\omega$',fontsize=14)
axs[0,0].set_ylabel(r'$log_{10}(D(\rho(t)||\rho_{ss}))$',fontsize=14)
axs[0,0].set_ylim(-7, 0.2)
axs[0,0].set_yticks(np.arange(-6, 0.1, 2))
axs[0,0].legend(fontsize=14)
axs[0,0].text(0.22, 0.98, '(a)', 
            transform=axs[0,0].transAxes,  # 使用轴坐标系统
            fontsize=14, 
            fontweight='bold',
            va='top', ha='left')
# 在右侧子图绘制数据
axs[0,1].plot(timelist, list1, color='blue',linestyle = '-',label=r'$p_\alpha$')
axs[0,1].plot(timelist, list2, color='green',linestyle = '-',label=r'$p_\beta$')
axs[0,1].set_xlabel(r'$t\omega$',fontsize=14)
axs[0,1].set_ylim(-7, 0.2)  # 设置y轴范围
axs[0,1].set_yticks(np.arange(-6, 0.1, 2))
axs[0,1].legend(fontsize=14)
axs[0,1].text(0.22, 0.98, '(b)', 
            transform=axs[0,1].transAxes,  # 使用轴坐标系统
            fontsize=14, 
            fontweight='bold',
            va='top', ha='left')

axs[1,0].plot(timelist, list3, color='blue',linestyle = '-',label=r'$p_\phi$')
axs[1,0].plot(timelist, list4, color='green',linestyle = '-',label=r'$p_\psi$')

axs[1,0].set_xlabel(r'$t\omega$',fontsize=14)
axs[1,0].set_ylim(-7, 0.2)  # 设置y轴范围
axs[1,0].set_yticks(np.arange(-6, 0.1, 2))
axs[1,0].set_ylabel(r'$log_{10}(D(\rho(t)||\rho_{ss}))$',fontsize=14)
axs[1,0].legend(fontsize=14)
axs[1,0].text(0.22, 0.98, '(c)', 
            transform=axs[1,0].transAxes,  # 使用轴坐标系统
            fontsize=14, 
            fontweight='bold',
            va='top', ha='left')
axs[1,1].plot(timelist, list3d, color='blue',linestyle = '-',label=r'$p_\phi$')
axs[1,1].plot(timelist, list4d, color='green',linestyle = '-',label=r'$p_\psi$')

axs[1,1].set_xlabel(r'$t\omega$',fontsize=14)
axs[1,1].set_ylim(-7, 0.2)  # 设置y轴范围
axs[1,1].set_yticks(np.arange(-6, 0.1, 2))
axs[1,1].legend(fontsize=14)
axs[1,1].text(0.22, 0.98, '(d)', 
            transform=axs[1,1].transAxes,  # 使用轴坐标系统
            fontsize=14, 
            fontweight='bold',
            va='top', ha='left')
# 调整子图间距
plt.tight_layout()


# 显示图形
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),dpi=400)  # 2行1列的子图

# 3. 在上子图绘制a和b
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(timelist, list1d, label=r'$p_\alpha,\Delta_1=0$', color='blue')
ax1.plot(timelist, list1, label=r'$p_\alpha,\Delta_1=0.4$', color='blue',linestyle='--')
ax1.plot(timelist, list2d, label=r'$p_\beta,\Delta_1=0$', color='red')
ax1.plot(timelist, list2,  label=r'$p_\beta,\Delta_1=0.4$',color='red',linestyle='--')
ax1.set_ylabel(r'$log_{10}(D(\rho(t)||\rho_{ss}))$',fontsize=20)
ax1.legend(fontsize=17)
ax1.text(-0.05, 0.99, '(a)', transform=ax1.transAxes, 
         fontsize=20, fontweight='bold', va='top', ha='right')


# 4. 在下子图绘制c和d
ax2.tick_params(axis='both', labelsize=20)
ax2.plot(timelist, list3, label=r'$p_\phi,\Delta_1=0$', color='green')
ax2.plot(timelist, list3d , label=r'$p_\phi,\Delta_1=0.4$', color='green',linestyle='--')
ax2.plot(timelist, list4, label=r'$p_\psi,\Delta_1=0$', color='purple')
ax2.plot(timelist, list4d, label=r'$p_\psi,\Delta_1=0.4$', color='purple',linestyle='--')
ax2.set_xlabel(r'$\omega_2t$',fontsize=20)
ax2.set_ylabel(r'$log_{10}(D(\rho(t)||\rho_{ss}))$',fontsize=20)
ax2.legend(fontsize=17)
ax2.text(-0.05, 0.99, '(b)', transform=ax2.transAxes, 
         fontsize=20, fontweight='bold', va='top', ha='right')

# 5. 调整子图间距并显示
plt.tight_layout()  # 防止标签重叠
plt.show()

