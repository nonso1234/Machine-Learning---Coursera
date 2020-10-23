%% Univariate Linear Regression 
%%%%%%%
%Run as separate scripts: Univariate and multivariate
%%%%%%%

clc; clear; close all 
%Load data 
data = load('ex1data1.txt'); 
%Separate into 
X = data(:,1); y = data(:,2); 
plotdata(X,y)
n = 2;
m = length(X); 

X = [ones(m,1), X]; %append ones 
theta = [0;0];
alpha = 0.02;
iternum = 1500; 

%Test cost function 
%computecost([-1;2],X,y,m)

%%%% Run gradient descent (One variable)%%%%%
[theta, j_hist] = gradient(X,y,m,alpha,theta, iternum,n);
fit  = theta(1) + theta(2)*X(:,2);

%Run prediction for 
predict1 = [1,3.5]*theta; 
fprintf('For population = 35,000, we predict a profit of %0.2f\n', predict1*10000);
predict2 = [1, 7]*theta; 
fprintf('Population of 70,000, predicted profit is %0.2f\n', predict2*10000); 

%Regression fit Plot 
hold on; 
plot(X(:,2),fit,'k', 'LineWidth',2)
title('Regression fit to predict profits based on population') 

%show cost reduction plot 
figure
plot(j_hist, '-', 'LineWidth', 2)
xlabel('Iterations')
ylabel('Cost') 
%Contour plots 

%% Multivariate Linear regression
%Predicting house prices from sq footage and no of bedrooms
clc; clear; close all 

%Import data 
data = load('ex1data2.txt');
X = data(:,1:2);
y = data(:,3); 
m = length(X);

%Gradient descent settings
theta = [0;0;0]; %
n = length(theta); 
alpha = 0.01; 
iternum = 500; 

[X,mu,stdev] = featureNorm(X, n-1); 

%add column of ones
X = [ones(m,1),X(:,1:2)];

%Run gradient descent 
[theta, j_hist] = gradient(X,y,m,alpha,theta, iternum,n); 

%Price Prediction for a 1650 sq ft 3Br house 
price = [1, ((1650 - mu(1))/stdev(1)), ((3 - mu(2))/stdev(2))]*theta; 
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%0.2f', price);

%Show cost reduction
plot(j_hist, '-', 'LineWidth', 2)
xlabel('No of iterations')
ylabel('Cost') 
title('Cost function Reduction') 



















