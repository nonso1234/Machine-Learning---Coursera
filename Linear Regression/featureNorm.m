
function[X,mu,stdev] = featureNorm(X,n)
%function normalizes the square foot data by subtracting the mean from each
%value and then dividing by the standard deviation 

mu = zeros(2,1);
stdev = zeros(2,1);
for i = 1:n
    mu(i) = mean(X(:,i));
    stdev(i) = std(X(:,i)); 
    X(:,i) = (X(:,i) - mean(X(:,i)))/ std(X(:,i));  
end 

    





