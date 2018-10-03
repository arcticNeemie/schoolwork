function new = newtonMethod(f,g,x0,tol)
%
% INPUT: f,g,x0,tol
%     f is the function
%     g is the derivative of f
%     x0 is the initial guess
%     tol is the tolerance
% OUTPUT: The approximate root of f
%
old = x0;
new = old - f(old)/g(old);
while abs(old-new)>tol
    old = new;
    new = old - f(old)/g(old);
end
end