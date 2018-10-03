x0 = [-1.7551;-4.0000];
tol = 0.0005;
maxIter = 1000;

n = 50;

R = linspace(0,0.5,n);

RS = [];
backtrack = 0.3;

K = [];
infCount = 0;
maxIterCount = 0;
globalMinCount = 0;
gm = [1;1];

minK = 1000;
mini = [0,0,0,0];
maxK = -1000;
maxi = mini;

for i=1:n
    S = linspace(R(i),1,n);
    for j=1:n
        i
        j
        newRS = [R(i);S(j)];
        RS = [RS,newRS];
        settings = [R(i),S(j),backtrack];
        [x,output] = quasiNewton(@Rosenbrock,@gradRosenbrock,x0,tol,maxIter,settings);
        if output.status == 0
            sK = output.iter;
            K = [K,sK];
            if sK < minK
                minK = sK;
                minRS = [R(i),S(j)];
            end
            if sK > maxK
                maxK = sK;
                maxRS = [R(i),S(j)];
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