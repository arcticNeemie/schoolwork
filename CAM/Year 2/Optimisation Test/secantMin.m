function approxMat = secantMin(f,g, x, n)
x0 = x(1);
x1 = x(2);
approxMat = [];
for i=1:n
    x2 = x1 - g(x1)*(x1-x0)/(g(x1)-g(x0));
    x0 = x1;
    x1 = x2;
    approxMat = [approxMat; x2,f(x2)];
end
end