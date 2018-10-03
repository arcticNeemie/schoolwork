%                        %
% Hyperparameter Testing %
%                        %

%HD Quadratic

radius = 5;
n = 1000;
X = linspace(-radius,radius,n);    %Generate n evenly spaced points between 1-radius and 1+radius

tol = 0.0005;
maxIter = 1000;
settings = [0.25,0.5,0.3];

K = [];
infCount = 0;
maxIterCount = 0;
globalMinCount = 0;
gm = zeros(100,1);

minK = 1000;
mini = 0;
maxK = -1000;
maxi = 0;

oneMat = ones(100,1);

K = [];

for i=1:n
    i
    x0 = X(i)*oneMat;
    [x,output] = fletcher(@HDquadratic,@gradHDquadratic,x0,tol,maxIter,settings);
    if output.status == 0
        sK = output.iter;
        K = [K,sK];
        if sK < minK
            minK = sK;
            mini = i;
        end
        if sK > maxK
            maxK = sK;
            maxi = i;
        end
    elseif output.status == 1
        maxIterCount = maxIterCount + 1;
    else
        infCount = infCount + 1;
    end
    if(norm(x-gm)<0.2)
        globalMinCount = globalMinCount + 1;
    end
end

