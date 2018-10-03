function [x,output] = fletcher(f,g,x,tol,maxIter,settings)
%
%Inputs
%   f - function to be minimised
%   g - gradient of function to be minimised
%   x - initial solution (column vector)
%   tol - tolerance value(for stopping)
%   maxIter - maximum number of iterations
%   settings - a vector of hyperparameters which looks like
%   [rho,sigma,backtrack]
%Outputs
%   x - optimal solution.
%   output. - matlab structure with the following fields
%       .iter - number of iterations.
%       .status - 0: gradient reached tolerance.
%                 1: reached maximum number of iterations.
%       .fstar - optimal function value f(x*).
%       .xHist - History of x i.e store the x at each iteration.
%       .fHist - History of f i.e store f at each iteration.
%

%Initialize Output
k = 0;  %Initialise iteration count
output.xHist = x;   %Initialise xHist
output.fHist = f(x);  %Initialise fHist

rho = settings(1); %initialising rho
sigma = settings(2); %initialising sigma
backtrack = settings(3); %initialising backtrack

d = -g(x);

while norm(g(x)) > tol && k<maxIter
    alpha = 1; % setting alpha
    while (f(x+alpha*d) > (f(x) + transpose(alpha*rho*g(x))*d)... %Strong Wolfe Condition
        && abs(transpose(g(x+alpha*d))*d)>sigma*abs(transpose(g(x))*d))
        alpha = alpha*backtrack; %using backtracking until alpha satisfies strong Wolfe
    end
    old = x;
    x = x + alpha*d;
    B = (norm(g(x))^(2))/(norm(g(old))^(2));
    d = -g(x)+B*d;
    k = k + 1;
    output.xHist = [output.xHist,x];
    output.fHist = [output.fHist,f(x)];
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