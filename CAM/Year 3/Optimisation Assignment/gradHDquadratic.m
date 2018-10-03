function grad = gradHDquadratic(x)
%
%Inputs
%   x: a 100 dimensional vector.
%
%Outputs
%   grad: the out grad for the HDquadratic function.
%
grad = zeros(100,1);
for i = 1:100
    grad(i) = 2*x(i)*i;
end 
end