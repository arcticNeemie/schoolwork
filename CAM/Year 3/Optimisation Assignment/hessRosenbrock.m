function H = hessRosenbrock(x)
%
%Inputs
%   x: a 2-dimensional vector
%
%Outputs
%   H: the Hessian Matrix for the Rosenbrock function.
%
H = [-400*x(2)+1200*(x(1)^2)+2,-400*x(1);-400*x(1),200];
end