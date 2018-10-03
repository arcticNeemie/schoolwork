function x1 = secantMethod(df, x, tol)
%
% INPUT: (df,x,tol)
%     df - the derivative function of the function to be minimised
%     x - a vector [x0,x1] of initial guesses
%     tol - a tolerance scalar
% OUTPUT: the x coordinate of the function to be minimised
%
x0 = x(1);
x1 = x(2);
while abs(x1-x0)>tol
    x2 = x1 - df(x1)*(x1-x0)/(df(x1)-df(x0));
    x0 = x1;
    x1 = x2;
end
end