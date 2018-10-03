function [A,R] = stirlingError(n)
%
%INPUT: n
%OUTPUT: A, R (vectors)
%A is the absolute error for each value of stirling's approximation less
%than n. R is for relative error
%
A = ones(1,n);
R = ones(1,n);
s = @(x) sqrt(2*pi*x)*((x/exp(1))^x);
for i=1:n
    A(i) = abs(factorial(i)-s(i));
    R(i) = A(i)/factorial(i);
end
end