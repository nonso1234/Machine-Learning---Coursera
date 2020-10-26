function [J,grad] = costFunctionReg(init_theta,X,y,m,lambda)
%Returns the cost function and gradient
z = X*init_theta;
hyp = sigmoid(z);
reg = (lambda/(2*m)) * sum(init_theta(2:end).^2);
%cost function 
J = (1/m)*sum(-y.*log(hyp)-(1-y).*log(1-hyp)) + reg;
%gradient 
grad = zeros(3,1); 
grad(1) = (1/m)*sum(hyp - y); 
for i = 2:length(init_theta)
    grad(i) = (1/m)*sum((hyp - y).*X(:,i)) + (lambda/m)*init_theta(i);

end 