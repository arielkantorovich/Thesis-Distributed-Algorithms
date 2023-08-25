# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:10:32 2020

@author: ilaib
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from numpy import random as rnd

# %% Setting

T = 2000;
N = 300;
K = 3;  # Number of hours

NumOfTests = 100;

Rewards = np.zeros((T, N, K));
SumRewards = np.zeros((T, N));

RewardsUncontrolled = np.zeros((T, N, K));
SumRewardsUncontrolled = np.zeros((T, N));

alpha = 0 * np.ones((NumOfTests, T, K));

StepSize = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.6));  # 0.1, 0.8
ControlStepSize = 0.8 * np.reciprocal(np.power(range(1, T + 1), 0.9));  # 10,0.8
delta = 0.25 * np.reciprocal(np.power(range(1, T + 1), 0.4));

gradients = np.zeros((N, K));
gradientsUncontrolled = np.zeros((N, K));

ConvergenceError = np.zeros((NumOfTests, T));
TotalLoads = np.zeros((NumOfTests, T, K));
TotalLoadsUncontrolled = np.zeros((NumOfTests, T, K));
SocialWelfare = np.zeros((NumOfTests, T));
SocialWelfareUncontrolled = np.zeros((NumOfTests, T));

N0 = 0.001;
beta = 2 / N;
ChannelGains = beta * rnd.rand(N, K, N);
Locations = np.zeros((N, 2));

SumEnergyConstraint = 1 + 0 * 0.5 * K;

TargetLoads = (N / 100) * np.array([20, 25, 30]);

# %% Iterations

for i in range(0, NumOfTests):
    Energy = 0.1 * np.ones((T, N, K));  # 10*rnd.rand(T,N,K);
    EnergyUncontrolled = 0.1 * np.ones((T, N, K));  # 10*rnd.rand(T,N,K);

    for n in range(0, N):
        Locations[n, :] = np.sqrt(2 * N) * rnd.rand(2);

    for n in range(0, N):
        for m in range(0, N):
            ChannelGains[n, :, m] = np.minimum(1 / np.power(LA.norm(Locations[n, :] - Locations[m, :]), 2),
                                               100000 * beta);

    for n in range(0, N):
        ChannelGains[n, :, n] = 2.5 + 3 * rnd.rand(K);

    for t in range(0, T - 1):
        for n in range(0, N):
            gradientNoise = delta[t] + 0.25 * rnd.randn(1, K);

            Interference = np.multiply(ChannelGains[:, :, n], Energy[t, :, :]).sum(0) - ChannelGains[n, :, n] * Energy[
                                                                                                                t, n,
                                                                                                                :];
            gradients[n, :] = ChannelGains[n, :, n] / (N0 + Interference) - alpha[i, t, :] + gradientNoise;
            Rewards[t, n, :] = np.log2(1 + ChannelGains[n, :, n] * Energy[t, n, :] / (N0 + Interference));
            SumRewards[t, n] = Rewards[t, n, :].sum();

            Interference = np.multiply(ChannelGains[:, :, n], EnergyUncontrolled[t, :, :]).sum(0) - ChannelGains[n, :,
                                                                                                    n] * EnergyUncontrolled[
                                                                                                         t, n, :];
            gradientsUncontrolled[n, :] = ChannelGains[n, :, n] / (N0 + Interference) + gradientNoise;
            RewardsUncontrolled[t, n, :] = np.log2(
                1 + ChannelGains[n, :, n] * EnergyUncontrolled[t, n, :] / (N0 + Interference));
            SumRewardsUncontrolled[t, n] = RewardsUncontrolled[t, n, :].sum();

        for n in range(0, N):
            Energy[t + 1, n, :] = Energy[t, n, :] + StepSize[t] * gradients[n, :];
            Energy[t + 1, n, :] = np.maximum(np.minimum(Energy[t + 1, n, :], 1000), 0);
            ProjectionFactor = np.minimum(SumEnergyConstraint / Energy[t + 1, n, :].sum(), 1);
            Energy[t + 1, n, :] = ProjectionFactor * Energy[t + 1, n, :];

            EnergyUncontrolled[t + 1, n, :] = EnergyUncontrolled[t, n, :] + StepSize[t] * gradientsUncontrolled[n, :];
            EnergyUncontrolled[t + 1, n, :] = np.maximum(np.minimum(EnergyUncontrolled[t + 1, n, :], 1000), 0);
            ProjectionFactor = np.minimum(SumEnergyConstraint / EnergyUncontrolled[t + 1, n, :].sum(), 1);
            EnergyUncontrolled[t + 1, n, :] = ProjectionFactor * EnergyUncontrolled[t + 1, n, :];

        alpha[i, t + 1, :] = alpha[i, t, :] + ControlStepSize[t] * (Energy[t, :, :].sum(0) - TargetLoads);
        alpha[i, t + 1, :] = np.maximum(alpha[i, t + 1, :], 0);

        for k in range(0, K):
            ConvergenceError[i, t] += np.abs((Energy[t, :, k].sum(0) - TargetLoads[k]) * alpha[i, t, k]);

        TotalLoads[i, t, :] = Energy[t, :, :].sum(0);
        SocialWelfare[i, t] = SumRewards[t, :].sum(0);

        TotalLoadsUncontrolled[i, t, :] = EnergyUncontrolled[t, :, :].sum(0);
        SocialWelfareUncontrolled[i, t] = SumRewardsUncontrolled[t, :].sum(0);

    display(i);

# %% Plots


meanSocialWelfare = np.mean(SocialWelfare, axis=0);
stdSocialWelfare = np.std(SocialWelfare, axis=0);

meanSocialWelfareUncontrolled = np.mean(SocialWelfareUncontrolled, axis=0);
stdSocialWelfareUncontrolled = np.std(SocialWelfareUncontrolled, axis=0);

plt.fill_between(range(0, T - 1), meanSocialWelfare[0:T - 1] - stdSocialWelfare[0:T - 1],
                 meanSocialWelfare[0:T - 1] + stdSocialWelfare[0:T - 1], alpha=0.5);
plt.fill_between(range(0, T - 1), meanSocialWelfareUncontrolled[0:T - 1] - stdSocialWelfareUncontrolled[0:T - 1],
                 meanSocialWelfareUncontrolled[0:T - 1] + stdSocialWelfareUncontrolled[0:T - 1], alpha=0.5);

SumewardsPlot, = plt.plot(range(0, T - 1), meanSocialWelfare[0:T - 1], '--', label='Sum of Rewards - Our algorithm')
SumewardsUncontrolledPlot, = plt.plot(range(0, T - 1), meanSocialWelfareUncontrolled[0:T - 1], '--',
                                      label='Sum of Rewards - Uncontrolled System')

plt.legend()

plt.xlabel("Iterations")
plt.ylabel("Sum of Rewards")
plt.grid(True)
plt.show()

meanTotalLoads = np.mean(TotalLoads, axis=0);
stdTotalLoads = np.std(TotalLoads, axis=0);

meanTotalLoadsUncontrolled = np.mean(TotalLoadsUncontrolled, axis=0);
stdTotalLoadsUncontrolled = np.std(TotalLoadsUncontrolled, axis=0);

ColorArray = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00'];

TargetLodasPlot, = plt.plot(range(0, T - 1), np.ones((T - 1)) * TargetLoads[0], 'k--',
                            label='Target Total Transmission Power')
EnergyPlots, = plt.plot(range(0, T - 1), meanTotalLoads[0:T - 1, 0], 'k',
                        label='Total Transmission Power: Our Algorithm')
EnergyUncontrolledPlot, = plt.plot(range(0, T - 1), meanTotalLoadsUncontrolled[0:T - 1, 0], color='k',
                                   linestyle='dotted', label='Total Transmission Power: Uncontrolled System')
plt.fill_between(range(0, T - 1), meanTotalLoads[0:T - 1, 0] - stdTotalLoads[0:T - 1, 0],
                 meanTotalLoads[0:T - 1, 0] + stdTotalLoads[0:T - 1, 0], color='k', alpha=0.5);
plt.fill_between(range(0, T - 1), meanTotalLoadsUncontrolled[0:T - 1, 0] - stdTotalLoadsUncontrolled[0:T - 1, 0],
                 meanTotalLoadsUncontrolled[0:T - 1, 0] + stdTotalLoadsUncontrolled[0:T - 1, 0], color='k', alpha=0.5);

for k in range(0, K):
    TargetLodasPlot, = plt.plot(range(0, T - 1), np.ones((T - 1)) * TargetLoads[k], color=ColorArray[k],
                                linestyle='dashed');
    EnergyPlots, = plt.plot(range(0, T - 1), meanTotalLoads[0:T - 1, k], color=ColorArray[k], linestyle='solid');
    plt.fill_between(range(0, T - 1), meanTotalLoads[0:T - 1, k] - stdTotalLoads[0:T - 1, k],
                     meanTotalLoads[0:T - 1, k] + stdTotalLoads[0:T - 1, k], color=ColorArray[k], alpha=0.5);
    EnergyUncontrolledPlot, = plt.plot(range(0, T - 1), meanTotalLoadsUncontrolled[0:T - 1, k], color=ColorArray[k],
                                       linestyle='dotted');
    plt.fill_between(range(0, T - 1), meanTotalLoadsUncontrolled[0:T - 1, k] - stdTotalLoadsUncontrolled[0:T - 1, k],
                     meanTotalLoadsUncontrolled[0:T - 1, k] + stdTotalLoadsUncontrolled[0:T - 1, k],
                     color=ColorArray[k], alpha=0.5);

plt.legend()
# plt.legend(loc="upper left")
plt.xlabel("Iterations")
plt.ylabel("Total Transmission Power")
plt.grid(True)
plt.show()

meanAlpha = np.mean(alpha, axis=0);
stdAlpha = np.std(alpha, axis=0);

for k in range(0, K):
    AlphaPlot = plt.plot(range(0, T - 1), meanAlpha[0:T - 1, k], color=ColorArray[k], label='k=' + str(k + 1))
    plt.fill_between(range(0, T - 1), meanAlpha[0:T - 1, k] - stdAlpha[0:T - 1, k],
                     meanAlpha[0:T - 1, k] + stdAlpha[0:T - 1, k], color=ColorArray[k], alpha=0.5);

# plt.legend(loc="upper left")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Alphas")
plt.grid(True)
plt.show()

meanError = np.mean(ConvergenceError, axis=0);
stdError = np.std(ConvergenceError, axis=0);
ErrorPlot = plt.plot(range(0, T - 1), meanError[0:T - 1], label='Convergence Error')
plt.fill_between(range(0, T - 1), meanError[0:T - 1] - stdError[0:T - 1], meanError[0:T - 1] + stdError[0:T - 1],
                 alpha=0.5);
# plt.legend(loc="upper left")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Convergence Error")
plt.grid(True)
plt.show()
