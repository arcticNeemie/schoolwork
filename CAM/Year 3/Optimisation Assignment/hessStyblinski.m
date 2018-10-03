function H = hessStyblinski(x)
%
%Inputs
%   x: a 4-dimensional vector
%
%Outputs
%   H: the Hessian Matrix for the Styblinski-Tang function.
%
H = eye(4);
for i=1:4
    H(i,i) = 6*x(i)^(2) - 16;
end
end