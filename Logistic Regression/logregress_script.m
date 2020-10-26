%% Implementation script 
%Prediction of student's chances of getting admitted using logistic
%regression 
clc; clear; close all
%Load data 
data = load('ex2data1.txt'); X = data(:,1:2); 
y = data(:,3); 

%Plot data 
%vizdata(X,y); 
% xlabel('Exam 1 score')
% ylabel('Exam 2 Score')
% legend('Admitted', 'Not Admitted')

init_theta = [0;0;0];
[m,n] = size(X); 
X = [ones(m,1),X]; 

%Cost function and gradient 
[J,grad] = costFunction(init_theta,X,y,m); 
%%%Using fminunc optimization function %%%
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y,m)), init_theta, options);

plotDecisionBoundary(theta,X,y); 

%%Prediction 
%for a student with exam 1 score of 45 and exam 2 score of 85 
%prob = sigmoid([1 45 85]*theta);
%fprintf("Probability of a student getting admitted with a scores of 45 and 85 is: %f\n'", prob)

p = predict(theta,X); 
fprintf('Train Accuracy: %0.3f\n', mean(double(p == y)) * 100);

%% Regularized Logistic regression 
%Predict if microchips passes inspection based on historical data 
clc; clear; close all 
data = load('ex2data2.txt'); 
X = data(:,1:2); y = data(:,3); 
 %vizdata(X,y)

X = mapFeature(X(:,1), X(:,2));

init_theta = ones(size(X, 2), 1);

% Set regularization parameter lambda
lambda = 100;
[m,n] = size(X); 

% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad] = costFunctionReg(init_theta, X, y,m,lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

%Run fminunc optimizer 
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 900);
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X, y,lambda,m)), init_theta, options);

plotDecisionBoundary(theta,X,y);

% Labels and Legend
legend('y = 1', 'y = 0', 'Decision boundary')
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

%Accuracy for training set 
p = predict(theta, X);

fprintf("Accuracy: %0.3f\n ", mean(double(p==y))*100);









