%                        %
% Hyperparameter Testing %
%                        %

%Styblinski Tang

radius = 5;
n = 10;
X = linspace(-2.9035-radius,-2.9035+radius,n);    %Generate n evenly spaced points between 1-radius and 1+radius

tol = 0.0005;
maxIter = 1000;
settings = [0.25,0.5,0.3];

K = [];
infCount = 0;
maxIterCount = 0;
globalMinCount = 0;
gm = [-2.9035;-2.9035;-2.9035;-2.9035];

minK = 1000;
mini = [0,0,0,0];
maxK = -1000;
maxi = mini;

%figure
for i=1:n
    for j=1:n
        for k=1:n
            for l=1:n
                x0 = [X(i);X(j);X(k);X(l)];
                i
                j
                k
                l
                [x,output] = fletcher(@Styblinski,@gradStyblinski,x0,tol,maxIter,settings);
                if output.status == 0
                    sK = output.iter;
                    K = [K,sK];
                    if sK < minK
                        minK = sK;
                        mini = [i,j,k,l];
                    end
                    if sK > maxK
                        maxK = sK;
                        maxi = [i,j,k,l];
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
        end
    end
end

