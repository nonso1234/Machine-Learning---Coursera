function g = sigmoid(z)
%function returns the sigmoid of a single value. Vectored input returns
%sigmoid, as a vector, of every value in input 
g = 1 ./(1+exp(-z));