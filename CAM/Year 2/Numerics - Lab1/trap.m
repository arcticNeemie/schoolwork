function integral = trap(f,a,b,h)
n = (b-a)/h;
%if Chad gives us n, h = (b-a)/n
integral = f(a);
for i=1:n-1
    integral = integral + 2*f(a+i*h);
end
integral = (h/2)*(integral + f(b));
end
