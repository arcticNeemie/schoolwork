%Finite Difference Heat Equation - Matrix Implementation
clc
close all

%Parameters
dx = 0.00625;
dt = 0.0000195313;
D = 1;
f = @(x) exp(-100*(x-(1/2)).^2);

x0 = 0;
x1 = 1;

t0 = 0;
t1 = 0.005;

b0 = 0;
b1 = 0;

%x
x = x0:dx:x1;
N = length(x);
%time
time = t0:dt:t1;
T = length(time);

u = transpose(f(x));
L1 = (D*dt)/(dx^2);
L2 = 1 - 2*L1;

A = zeros(N,N);
A(1,1) = 1;
A(end,end) = 1;
for i=2:N-1
    A(i,i) = L2;
    A(i,i-1) = L1;
    A(i,i+1) = L1;
end

for n = 1:T
    old = u;
    u = A*old;
    %Plot
    figure(1);
    plot(x,u);
    axis([0,1,0,1])
    pause(0.001);
end

hold off

