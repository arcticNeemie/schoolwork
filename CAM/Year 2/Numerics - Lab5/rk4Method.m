function Tab = rk4Method(f,x0,y0,h,xf)
n = ((xf-x0)/h)+1;
Tab = [y0;zeros(n-1,1)];
col = transpose(x0:h:xf);
Tab = [col Tab];
for i=2:n;
    k1 = h*f(Tab(i-1,1),Tab(i-1,2));
    k2 = h*f(Tab(i-1,1)+h/2,Tab(i-1,2)+k1/2);
    k3 = h*f(Tab(i-1,1)+h/2,Tab(i-1,2)+k2/2);
    k4 = h*f(Tab(i-1,1)+h,Tab(i-1,2)+k3);
    Tab(i,2) = Tab(i-1,2) + (k1+2*k2+2*k3+k4)/6;
end
end