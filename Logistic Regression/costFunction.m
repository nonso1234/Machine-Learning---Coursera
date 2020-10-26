function [J,grad] = costFunction(init_theta,X,y,m)
%Function calculates the cost function and gradient 
z = X*init_theta;
hyp = sigmoid(z);
%cost function 
J = (1/m)*sum(-y.*log(hyp)-(1-y).*log(1-hyp));
%gradient 
grad = zeros(3,1); 
grad(1) = (1/m)*sum(hyp - y); 

for i = 2:length(init_theta)
    grad(i) = (1/m)*sum((hyp - y).*X(:,i));
end 




