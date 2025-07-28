# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:14:48 2025

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm

# 用于生成单条coherence曲线的函数
def compute_coherence_curve(v, Delta1, timescale=14, initial_state1=None, initial_state2=None,
                            r_21=1, r_32=1, r_31=8, w_2=1, w_3=2,
                            N_21=0, N_31=0, N_32=0):
    # 基础参数
    g1 = -0.5 * r_21 * (1 + 2 * N_21) - 0.5 * r_31 * N_31 - 0.5 * r_32 * N_32 + 1j * w_2
    g2 = -0.5 * r_31 * (1 + 2 * N_31) - 0.5 * r_32 * (1 + N_32) - 0.5 * r_21 * N_21 + 1j * w_3
    g3 = -0.5 * r_21 * (1 + N_21) - 0.5 * r_31 * (1 + N_31) - 0.5 * r_32 * (1 + 2 * N_32) - 1j * (w_2 - w_3)
    g4 = -0.5 * r_21 * (1 + N_21) - 0.5 * r_31 * (1 + N_31) - 0.5 * r_32 * (1 + 2 * N_32) - 1j * (w_3 - w_2)
    W0 = np.zeros((7, 7), dtype=complex)
    W0[0][0]=-r_31*N_31-r_21*N_21;W0[0][1]=r_21*(1+N_21);W0[0][2]=r_31*(1+N_31)
    W0[1][0]=r_21*N_21; W0[1][1]=-r_21*(1+N_21)-r_32*N_32;W0[1][2]=r_32*(1+N_32);
    W0[2][0]=r_31*N_31;W0[2][1]=r_32*N_32;W0[2][2]=-r_32*(1+N_32)-r_31*(1+N_31);
    W0[3][3]=g1;W0[4][4]=g2;W0[5][5]=g3;W0[6][6]=g4;
    I = np.eye(7, dtype=complex)
    Projection = np.hstack([I, np.zeros((7,7),dtype=complex)])

    Delta = np.zeros((7, 7), dtype=complex)
    Delta[1][5]=1;Delta[1][6]=-1
    Delta[2][5]=-1;Delta[2][6]=1
    Delta[3][4]=Delta[4][3]=Delta[5][1]=Delta[6][2]=1
    Delta[6][1]=Delta[5][2]=-1
    Delta = Delta * 1j * Delta1

    W = np.block([
        [W0,         Delta],
        [Delta, W0 - v * I]
    ])
    zero_vec = np.zeros(7)
    if initial_state1 is None:
        stat1 = np.array([0.3, 0.7, 0, 0, 0, 0, 0])
        state1 = np.concatenate([stat1, zero_vec])
    else:
        state1 = np.concatenate([initial_state1, zero_vec])
    if initial_state2 is None:
        stat2 = np.array([0.75, 0, 0.25, 0, np.sqrt(3)/4, 0, 0])
        state2 = np.concatenate([stat2, zero_vec])
    else:
        state2 = np.concatenate([initial_state2, zero_vec])

    eigenvalues, eigenvectors = np.linalg.eig(W)
    for i in range(len(eigenvalues)):
        if np.linalg.norm(eigenvalues[i]) < 1e-10:
            MaxE = i
            break
    VecA2 = np.linalg.inv(eigenvectors) @ state2
    eigenvectors = np.transpose(eigenvectors)
    fs = eigenvectors[MaxE]
    fs1 = Projection @ fs

    rho2 = np.zeros((3, 3), dtype=complex)
    rho_diag = np.zeros((3, 3), dtype=complex)
    times = []
    curve = []
    Npoints = timescale * 1000
    for i in range(Npoints):
        t = timescale * i / Npoints
        times.append(t)
        State2n = np.zeros(14, dtype=complex)
        for j in range(len(eigenvalues)):
            if j == MaxE:
                State2n += fs
            else:
                State2n += VecA2[j] * np.exp(eigenvalues[j] * t) * eigenvectors[j]
        State2n = Projection @ State2n
        for j in range(3):
            rho2[j, j] = State2n[j]
        rho2[0, 1] = State2n[3]
        rho2[0, 2] = State2n[4]
        rho2[1, 2] = State2n[5]
        rho2[1, 0] = np.conj(rho2[0, 1])
        rho2[2, 0] = np.conj(rho2[0, 2])
        rho2[2, 1] = np.conj(rho2[1, 2])
        for j in range(3):
            rho_diag[j, j] = State2n[j]
        # 相对熵
        de = np.trace(rho2 @ logm(rho2)) - np.trace(rho_diag @ logm(rho_diag))
        curve.append(np.log10(de))
    return np.array(times), np.array(curve)

# 曲线参数设定
param_list = [
    # (v, Delta1, label, color, linestyle)
    (0.1, 0.0, r'$\nu=0.1,\,\Delta_1=0$', 'blue', '-'),            # 蓝线
    (0.1, 0.2, r'$\nu=0.1,\,\Delta_1=0.2$', 'g', '-'),           # 绿线
    (0.05, 0.2, r'$\nu=0.05,\,\Delta_1=0.2$', 'red', '--'),        # 红虚线  
    (0.1, 0.5, r'$\nu=0.1,\,\Delta_1=0.5$', 'purple', '-.'),     # 紫色点划线
]

plt.figure(figsize=(7, 4), dpi=500)
for v, Delta1, label, color, linestyle in param_list:
    t, C = compute_coherence_curve(v, Delta1)
    plt.plot(t, C, color=color, linestyle=linestyle, label=label)

plt.xticks(np.arange(0, 12, 1))
plt.xlim(0,11)
plt.ylim(-9, 0.1)
plt.xlabel(r'$\omega_2t$', fontsize=13)
plt.ylabel(r'$\log_{10}\left(\mathcal{C}(\rho^{\beta}(t))\right)$', fontsize=13)
plt.tick_params(axis='y', labelsize=13)
plt.tick_params(axis='x', labelsize=13)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

