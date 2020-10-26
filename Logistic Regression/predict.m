function p = predict(theta,X)
%prediction in 0 and 1
m = size(X,1); 
p = zeros(m,1); 

for i = 1:m
    if sigmoid(X(i,:)*theta) < 0.5 
        p(i) = 0; 
    elseif sigmoid(X(i,:)*theta) >= 0.5
        p(i) = 1; 
    end 
end 
