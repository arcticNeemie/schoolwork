x = 1:64;
params = [ 1.93776595e-08,1.33297850e+00,2.96030296e+00,-7.64634615e+01,1.96906848e+04];
a = params(1);
b = params(2);
c = params(3);
d = params(4);
e = params(5);
f = @(xs) a*(b.^(c.*xs + d)) + e;
y = f(x);
plot(x,y)