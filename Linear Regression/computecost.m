function[J] = computecost(theta,X,y,m)
%Function computes the cost function 
%Input: theta, data X and Y, length of data m
hyp = X*theta; 
%Cost function 
J = 1/(2*m)*sum((hyp-y).^2); 





