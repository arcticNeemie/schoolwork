function x = newtonsMethod(F, J, x0, tol)
%
% INPUT: 
%   F   : n-dimensional function
%   J   : n x n Jacobian of F
%   x0  : column vector of n initial guesses
%   tol : scalar tolerance
%
% OUTPUT: The approximate roots of each equation in F
%   x   : an n-dimensional vector of the roots of F
%

old = x0;
x = old - inv(J(old))*F(old);
while max(abs(x-old))>tol
    old = x;
    x = old - inv(J(old))*F(old);
end
end

