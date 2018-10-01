clc
close all
clear all


f = @(x) sin(4*x);
xs = @(x) x;

x = -pi:0.01:pi;

m = 4;
A0 = (1/pi)*integral(f,-pi,pi)
g = (A0/2)*ones(1,length(x));
hold on
plot(x,f(x));
for i=1:m
    fcos = @(xs) f(xs).*cos(i*xs);
    fsin = @(xs) f(xs).*sin(i*xs);
    An = (1/pi)*integral(fcos,-pi,pi);
    Bn = (1/pi)*integral(fsin,-pi,pi);
    g = g + An*cos(i*x) + Bn*sin(i*x);
end
plot(x,g);
hold off
