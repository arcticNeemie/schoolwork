function integral = simpson(f,a,b,h)
n = (b-a)/h;
F = ones(1,n+1);
F(1) = f(a);
for i=1:n
    F(i+1) = f(a + i*h);
end
sum = 0;
for i=1:2:n-1
    sum = sum + F(i)+4*F(i+1)+F(i+2);
end
integral = (h/3)*sum;
end

