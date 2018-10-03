function [afinal,bfinal] = falsePosition(f, ainitial, binitial, tol)
%
% INPUT: f, ainitial, binitial, tol
%     f is the function
%     ainitial and binitial are endpoints of the search interval
%     tol is the tolerance
% OUTPUT: the final values for the endpoints of the search interval
%
a = ainitial;
b = binitial;
%Initial
g = @(x,y) (x*f(y)-y*f(x))/(f(y)-f(x));
cs = g(a,b);
if f(a)*f(cs)<0
        b = cs;
    else
        a = cs;
end
c = g(a,b);
while abs(c-cs)>tol
    if f(a)*f(c)<0
        b = c;
    else
        a = c;
    end
    cs = c;
    c = g(a,b);
end
afinal = a;
bfinal = b;
end
