function Tab = eulersMethod(f, x0, y0, h, xf)
n = ((xf-x0)/h)+1;
Tab = [y0;zeros(n-1,1)];
x= x0:h:xf;
col = transpose(x);
Tab = [col Tab];
for i=2:n
    Tab(i,2) = Tab(i-1,2) + h*f(Tab(i-1,1),Tab(i-1,2));
end
end