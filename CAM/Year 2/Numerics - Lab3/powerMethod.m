function [l,y] = powerMethod(A,x0,tol)
%
% INPUT: A - n*n matrix
%        x0 - initial guess for eigenvector (size n)
%        tol - tolerance scalar
% OUTPUT: [l,y] - l is the dominant eigenvalue, y the corresponding eigenvector
%
y = x0*(1/norm(x0));
x = A*y;
yold = y;
y = x*(1/norm(x));
while(max(abs(y-yold))>tol)
    x = A*y;
    yold = y;
    y = x*(1/norm(x)); 
end
l = (dot(A*y,y)/dot(y,y));
end