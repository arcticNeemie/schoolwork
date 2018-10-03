function H = hessHDquadratic(~)
%
%Inputs
%   None, as the hessian does not depend on x
%
%Outputs
%   H: the Hessian Matrix for the HDquadratic function.
%
H = eye(100);
for i=1:100
    H(i,i) = 2*i;
end
end