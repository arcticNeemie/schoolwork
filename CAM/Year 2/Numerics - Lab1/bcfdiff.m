function df = bcfdiff(f,x,h)
df = ones(1,3);
df(1) = (f(x)-f(x-h))/h;
df(2) = (f(x+h)-f(x-h))/(2*h);
df(3) = (f(x+h)-f(x))/h;
end

