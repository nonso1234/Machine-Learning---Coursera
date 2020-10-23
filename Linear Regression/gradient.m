%Gradient descent optimizer 
function[theta, j_hist]= gradient(X,y,m,alpha,theta, iternum, n)
%Gradient descent cost function 
%Input: X and Y data, learning rate, theta, n(no of features)
%Output: optimized theta after iteration is exhausted and cost history
j_hist = zeros(m,1);
for i = 1:iternum
    hyp = X*theta;
    theta(1) = theta(1) - (alpha*(1/m)*sum(hyp - y));
    for j = 2:n
        theta(j) = theta(j) - (alpha*(1/m)*sum((hyp - y).*X(:,j)));
    end 
    j_hist(i) = computecost(theta,X,y,m);
end 





