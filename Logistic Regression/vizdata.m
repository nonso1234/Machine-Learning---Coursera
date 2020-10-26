function[] = vizdata(X,y)
%Find indices of y = 1 and y = 0 in y array 
pos = find(y ==1); neg = find(y == 0);

plot(X(pos,1),X(pos,2),'k+', 'LineWidth', 1,'MarkerSize', 5)
hold on
plot(X(neg,1), X(neg,2),'ko', 'MarkerFaceColor', 'b','MarkerSize', 5)

% Specified in plot order



