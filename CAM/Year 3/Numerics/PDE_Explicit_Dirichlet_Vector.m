%Dirichlet, Explicit, Vector
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

%Initial Condition
u = transpose(f(x));

%Boundary Condition
u(1) = b0;
u(end) = b1;

R = D*dt/(dx^2);

%Iterate across time
for t=1:T
    old = u;
    u(2:end-1) = old(2:end-1) + R*(u(3:end)-2*(u(2:end-1))+u(1:end-2));
    
    %Plot
    figure(1);
    plot(x,u);
    axis([x0,x1,0,1])
    pause(0.001);
end

hold off