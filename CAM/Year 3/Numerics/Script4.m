%Finite Difference Heat Equation - Matrix Implementation
clc
close all


f = @(x) exp(-100*(x-(1/2)).^2);
N = 100;
xs = linspace(0,1,N);
plot(xs,f(xs));

D = 1;              %coefficient of diffusion
dx = 1/(N-1);
dt = (dx^2)/(2*D);
T = round(1/(dt)); %final time

u = transpose(f(xs));
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
    plot(xs,u);
    axis([0,1,0,1])
    pause(0.001);
end

hold off

