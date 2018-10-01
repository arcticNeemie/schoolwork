%Finite Difference Heat Equation - Vector Implementation
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

u = f(xs);

for n = 1:T
    old = u;
    u(2:end-1) = old(2:end-1) + ((D*dt)/(dx^2))*(old(3:end)-2*old(2:end-1)+old(1:end-2));
    %Plot
    figure(1);
    plot(xs,u);
    axis([0,1,0,1])
    pause(0.001);
end

hold off