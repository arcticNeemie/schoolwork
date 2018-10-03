function [x,output] = quasiNewton(F,J,x0,tol,maxIter,settings)
%
%Inputs
%   f - function to be minimized.
%   J - gradient of the function.
%   x0 - initial solution, column vector
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
%                 2: x has been set to NaN due to division by gamma = 0
%       .fstar - optimal function value f(x*).
%       .xHist - History of x i.e store the x at each iteration.
%       .fHist - History of f i.e store f at each iteration.
%
H = eye(length(x0)); %initialising Hessian as identity matrix
k = 0; %iteration 1 (first iteration)
rho = settings(1); %initialising rho
sigma = settings(2); %initialising sigma
backtrack = settings(3); %initialising backtrack
output.xHist = transpose(x0); % initialising history of x
output.fHist = F(x0); % initialising history of y
while norm(J(x0)) > tol && k<maxIter%check if every value is less than tolerance
    d = -H*J(x0); % setting d - First Step
    alpha = 1; % setting alpha
    while (F(x0+alpha*d) > (F(x0) + transpose(alpha*rho*J(x0))*d)... %Strong Wolfe Condition
        && abs(transpose(J(x0+alpha*d))*d)>sigma*abs(transpose(J(x0))*d))
        alpha = alpha*backtrack; %using backtracking until alpha satisfies strong Wolfe
    end
    xnew = x0 + alpha*d; %setting new x value
    del = xnew - x0; % setting del
    gamma = J(xnew) - J(x0); % setting gamma
    if gamma==0
        output.fstar = F(x0);
        output.iter = k;
        output.status = 2;
        x = x0;
        return
    end
    v = del/(transpose(del)*gamma) - (H*gamma)/(transpose(gamma)*H*gamma); %finding v
    Hchange = (del*transpose(del))/(transpose(del)*gamma); %first part Hchange
    Hchange = Hchange - (H*gamma*transpose(H*gamma))/(transpose(gamma)*H*gamma); %second part Hchange
    Hchange = Hchange + transpose(gamma)*H*gamma*v*transpose(v);
    H = H + Hchange; %updating H
    k = k+1 ; %Updating Step Size
    x0 = xnew; % setting x0 to the new x value we found
    output.xHist = [output.xHist ; transpose(x0)]; %adding value to x History
    output.fHist = [output.fHist ; F(x0)]; %adding value to y History
end
if norm(J(x0)) <= tol
    output.status = 0;
else
    output.status = 1;
end
output.fstar = F(x0);
output.iter = k;

x = x0; % Setting the output of the function

end