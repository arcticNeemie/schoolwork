function grad = gradStyblinski(x)
%
%Inputs
%   x: a 4 dimensional vector.
%
%Outputs
%   grad: the out grad for the Styblinski-Tang function.
%
grad = zeros(4,1);
for i = 1:4
    grad(i) = 0.5*(4*x(i)^(3) - 32*x(i) + 5);
end 
end