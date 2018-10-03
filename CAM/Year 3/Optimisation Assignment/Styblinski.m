function f = Styblinski(x)
%
%Inputs
%   x: a 4 dimensional vector.
%
%Outputs
%   f: the output value of the function.
%
f = 0 ;
for i= 1:4
    f = f + 0.5*(x(i)^4-16*x(i)^2+5*x(i)) ;
end
end