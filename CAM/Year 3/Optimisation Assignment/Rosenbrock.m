function f = Rosenbrock(x)
%
%Inputs
%   x: a 2 dimensional vector.
%
%Outputs
%   f: the output value of the function.
%
f = 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
end