function [x,y] = goldenSearch(f,a,b,tol);
% 
% INPUT: f,a,b,tol
%     f - the function to be minimised
%     a - the lower endpoint
%     b - the upper endpoint
%     tol - the tolerance
% OUTPUT: [x,y]
%     x - the x coordinate of the minimum of f
%     y - f(x)
% 
p = (3-sqrt(5))/2;
anew = a + p*(b-a);
bnew = b - p*(b-a);

while abs(bnew-anew)>tol
    if f(anew)<f(bnew)
        b = bnew;
    else
        a = anew;
    end
    anew = a + p*(b-a);
    bnew = b - p*(b-a);
end
x = 0.5*(anew+bnew);
y = f(x);
end