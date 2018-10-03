function f = HDquadratic(x)
%
%Inputs
%   x: a 100 dimensional vector.
%
%Outputs
%   f: the output value of the function.
%
f = 0 ;
for i= 1:100
    f = f + i*x(i)^2 ;
end
end