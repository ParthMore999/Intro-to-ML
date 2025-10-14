%% Question 2: Setup
clear all; close all; clc;

priors = [0.25, 0.25, 0.25, 0.25];

mu1 = [-2; -2];
mu2 = [2; -2];
mu3 = [2; 2];
mu4 = [-2; 2];

Sigma1 = [0.8, 0.2; 0.2, 0.5];
Sigma2 = [0.5, -0.3; -0.3, 0.8];
Sigma3 = [0.6, 0; 0, 0.6];
Sigma4 = [1.0, 0.4; 0.4, 0.3];

save('q2_parameters.mat', 'priors', 'mu1', 'mu2', 'mu3', 'mu4', ...
     'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4');
