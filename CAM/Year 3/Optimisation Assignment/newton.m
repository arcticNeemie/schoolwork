function [x,output] = newton(f,g,J,x0,tol,maxIter)
%
%Inputs
%   f - function to be minimised
%   g - gradient of function to be minimised
%   J - hessian matrix of the function to be minimised
%   x0 - initial solution (column vector)
%   tol - tolerance value(for stopping)
%   maxIter - maximum number of iterations
%Outputs
%   x - optimal solution.
%   output. - matlab structure with the following fields
%       .iter - number of iterations.
%       .status - 0: gradient reached tolerance.
%                 1: reached maximum number of iterations.
%       .fstar - optimal function value f(x*).
%       .xHist - History of x i.e store the x at each iteration.
%       .fHist - History of f i.e store f at each iteration.
%       .gHist - History of g i.e store g at each iteration.
%


%Initialize Output
k = 0;  %Initialise iteration count
output.xHist = x0;   %Initialise xHist
output.fHist = f(x0);  %Initialise fHist
output.gHist = g(x0);    %Initialise gHist
x = x0;

while norm(g(x))>tol && k<maxIter
    old = x;
    x = old - inv(J(old))*g(old);
    k = k+1;
    output.xHist = [output.xHist,x];
    output.fHist = [output.fHist,f(x)];
    output.gHist = [output.gHist,g(x)];
end

%Check for end condition
if norm(g(x))<=tol
    output.status = 0;
else
    output.status = 1;
end

output.fstar = f(x);
output.iter = k;

end