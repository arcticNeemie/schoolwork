function g = gradRosenbrock(x)
%
%Inputs
%   x: a 2 dimensional vector.
%
%Outputs
%   g: the output value of the function.
%
g = [200*(x(2)-x(1)^2)*(-2*x(1))-2*(1-x(1));200*(x(2)-x(1)^2)];

end