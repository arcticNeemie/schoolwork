function [afinal,bfinal, i] = bisectionSearch(f, ainitial, binitial, tol)
%
% INPUT: f, ainitial, binitial, tol
%     f is the function
%     ainitial and binitial are endpoints of the search interval
%     tol is the tolerance
% OUTPUT: the final values for the endpoints of the search interval
%
a = ainitial;
b = binitial;
i = 0;
while abs(b-a)>tol
    c = 0.5*(a+b);
    i=i+1;
    if f(a)*f(c)<0
        b = c;
    else
        a = c;
    end
end
afinal = a;
bfinal = b;
end
