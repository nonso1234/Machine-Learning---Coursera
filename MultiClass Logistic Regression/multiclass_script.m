%% Multi-class Regression - Script 1
%Run as separate scripts. Use the Run section tab 
%The goal of this exercise is to use multiclass (one v all) logistic
%regression to predict handwritten digits from 0-9

clear; clc; close all
load('ex3data1.mat');
m = size(X, 1);

%Visualize Data: Digit grid 
% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

%Testing cost function 
%init_theta = zeros(size(X,2),1);
%[J, grad] = costFunction(init_theta,X,y,m);

%%%%%%Test gradient function : Uncomment to test gradient and cost function
%%%%%%with test inputs 
% init_theta = [-2; -1; 1; 2];
% X_t = [ones(5,1) reshape(1:15,5,3)/10];
% y_t = ([1;0;1;0;1] >= 0.5);
% lambda_t = 3;
% m_t = size(X_t,1);
% [J, grad] = lrcostFunction(init_theta,X_t,y_t,m_t,lambda_t);
% fprintf('Cost: %f | Expected cost: 2.534819\n',J);
% fprintf('Gradients:\n'); fprintf('%f\n',grad);
% fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003');
%%%%%%%

num_labels = 10; % 10 labels, from 1 to 10 
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

[pred,psig] = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%% Script 2: Neural networks 
%Using Neural network pre trained parameters to predict digits 0-9 
clear; clc; close all
load('ex3data1.mat') 
load('ex3weights.mat') %load pre-trained parameters 

[pmax,pred] = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

m = size(X,1);
%  Randomly permute examples
rp = randi(m);

%Predict 
pred = predict(Theta1, Theta2, X(rp,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
% Display 
displayData(X(rp, :));   

