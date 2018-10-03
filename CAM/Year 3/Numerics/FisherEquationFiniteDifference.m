%Fisher Equation - Dirichlet

clc
close all


f = @(x) 1./((exp(10*(sqrt(10/3)).*x)+1).^2);

dx = 0.00625;
dt = 0.0000195313;
N = (1/dx);
T = round(0.5/(dt));
p = 2000;

xs = linspace(0,1,N);
plot(xs,f(xs));

u = f(xs);
u(1) = 1;
u(N) = 0;

for n = 1:T
    old = u;
    for i=2:N-1
        u(i) = old(i) + ((dt)/(dx^2))*(old(i+1)-2*old(i)+old(i-1));
        u(i) = u(i)+ p*dt*old(i)*(1-old(i));
    end
    %Plot
    figure(1);
    plot(xs,u);
    axis([0,1,0,1])
    pause(0.01);
end

hold off