function ystar = LagrangeInterpolation(x,y,xstar)
%
%INPUT: x,y - vectors of size n
%INPUT: xstar - scalar
%OUTPUT: ystar - scalar
%
n = length(x);
ystar = 0;
for k=1:n
    product = 1;
    for i=1:n
        if i~=k
            product = product*(xstar-x(i))/(x(k)-x(i));
        end
    end
    ystar = ystar + product*y(k);
end
end