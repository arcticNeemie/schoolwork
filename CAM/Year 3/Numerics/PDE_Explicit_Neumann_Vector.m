%Neumann, Explicit, Vector
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

%Ghost Points
gpm1 = u(2)-2*dx*b0;        %May be u(1), not too sure
gpNp1 = u(end-1)+2*dx*b1;   %May be u(end), again, not too sure

u = [gpm1;u;gpNp1];

R = D*dt/(dx^2);

%Iterate across time
for t=1:T
    old = u;
    u(1) = u(3) - 2*dx*b0;          %Again, these could be u(2)
    u(end) = u(end-2) + 2*dx*b1;    %and u(end-1) respectively
    u(2:end-1) = old(2:end-1) + R*(u(3:end)-2*(u(2:end-1))+u(1:end-2));
    
    %Plot
    figure(1);
    plot(x,u(2:end-1));
    axis([x0,x1,0,1])
    pause(0.001);
end

hold off