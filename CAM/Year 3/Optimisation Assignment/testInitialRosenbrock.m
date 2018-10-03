%                        %
% Hyperparameter Testing %
%                        %

%BFGS
%We begin by testing the Rosenbrock function for various initial x's

radius = 5;
n = 50;
X = linspace(1-radius,1+radius,n);    %Generate n evenly spaced points between 1-radius and 1+radius

tol = 0.0005;
maxIter = 1000;
settings = [0.0612,0.0612,0.3];

K = zeros(n,n);
Q1 = K;
Q2 = K;
infCount = 0;
maxIterCount = 0;
globalMinCount = 0;

%figure
for i=1:n
    for j=1:n
        x0 = [X(i);X(j)];
        i
        j
        [x,output] = steepest(@Rosenbrock,@gradRosenbrock,x0,tol,maxIter,settings);
        if output.status == 0
            K(i,j) = output.iter;
        elseif output.status == 1
            K(i,j) = NaN;
            maxIterCount = maxIterCount + 1;
        else
            K(i,j) = NaN;
            infCount = infCount + 1;
        end
        Q1(i,j) = x(1);
        Q2(i,j) = x(2);
        if(x(1)<1.5 && x(1)>0.5 && x(2)<1.5 && x(2)>0.5)
            globalMinCount = globalMinCount + 1;
        end
    end
end

[min1,mini1] = min(K);
[min2,mini2] = min(min(K));
minPoint = [X(mini1(mini2));X(mini2)];
minIter = min2;

[max1,maxi1] = max(K);
[max2,maxi2] = max(max(K));
maxPoint = [X(maxi1(maxi2));X(maxi2)];
maxIter = max2;

Kvec1 = reshape(K,[1,n*n]);
Kvec = [];
for i=1:n*n
    if ~isnan(Kvec1(i))
        Kvec = [Kvec,Kvec1(i)];
    end
end



