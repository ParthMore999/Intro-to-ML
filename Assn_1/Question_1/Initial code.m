clear all, close all,

N = 10000;
p0 = 0.65; 
p1 = 0.35;

% Generate random class assignments based on priors
u = rand(1,N) >= p0;
N0 = length(find(u==0));
N1 = length(find(u==1));

% Generate Class 0 samples
mu = [-1/2; -1/2; -1/2];
Sigma = [1, -0.5, 0.3; 
         -0.5, 1, -0.5; 
         0.3, -0.5, 1];
r0 = mvnrnd(mu, Sigma, N0);

% Plot Class 0 samples in blue
figure(1), 
plot3(r0(:,1), r0(:,2), r0(:,3), '.b');
axis equal, hold on,

% Generate Class 1 samples
mu = [1; 1; 1];
Sigma = [1, 0.3, -0.2; 
         0.3, 1, 0.3; 
         -0.2, 0.3, 1];
r1 = mvnrnd(mu, Sigma, N1);

% Plot Class 1 samples in red
plot3(r1(:,1), r1(:,2), r1(:,3), '.r');
axis equal, hold on,

% Add labels for clarity
xlabel('X1'), ylabel('X2'), zlabel('X3')
title('3D Binary Classification Data')
legend('Class 0', 'Class 1')
grid on

% Save the data for future use
data = [r0; r1];  % Combine all samples
labels = [zeros(N0,1); ones(N1,1)];  % True class labels
save('classification_data.mat', 'data', 'labels', 'r0', 'r1', 'N0', 'N1');
