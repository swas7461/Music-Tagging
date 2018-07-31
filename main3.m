function [i] = main3(x)
x = [ones(size(x,1)), x];
theta = csvread('theta.csv');
h = sigmoid(x*theta');
[p, i] = max(h, [], 2);
end