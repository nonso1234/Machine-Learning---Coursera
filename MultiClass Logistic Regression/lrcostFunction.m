function [J,grad] = lrcostFunction(init_theta,X,y,lambda)
%Function calculates the cost function and gradient 
%Function calculates the cost function and gradient 
m = size(X,1);
z = X*init_theta;
hyp = sigmoid(z);
reg = (lambda/(2*m)) * sum(init_theta(2:end).^2);
%cost function 
J = (1/m)*sum(-y.*log(hyp)-(1-y).*log(1-hyp)) + reg;

%Vectorized gradient with regularization 
beta = hyp - y; 
grad1 = (1/m)*sum(beta .* X(:,1)); 
grad2 = (1/m)*sum(beta.*X(:,2:end))' + ((lambda/m) * init_theta(2:end));
grad = [grad1;grad2]; 




