function Tab = eulersModMethod(f, x0, y0, h, xf)
n = ((xf-x0)/h)+1;
Tab = [y0;zeros(n-1,1)];
col = transpose(x0:h:xf);
Tab = [col Tab];
for i=2:n
    yhalf = Tab(i-1,2) + h*f(Tab(i-1,1),Tab(i-1,2))/2;
    xhalf = Tab(i-1,1)+h/2;
    Tab(i,2) = Tab(i-1,2) + h*f(xhalf,yhalf);
end
end