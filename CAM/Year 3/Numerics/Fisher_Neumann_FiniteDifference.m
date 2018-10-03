%Fisher Equation - Neumann
clc
close all

p = 2000;
f = @(x) 1./((exp(10*sqrt(10/3).*x)+1).^2);
sol = @(x,t) 1./((exp(sqrt(p/6).*x-(5*p/6)*t)+1).^2);

ex = 4;

switch ex
    case 1
        dx = 0.05;
        dt = 0.00125;
    case 2
        dx = 0.025;
        dt = 0.0003125;
    case 3
        dx = 0.0125;
        dt = 0.000078125;
    case 4
        dx = 0.00625;
        dt = 0.0000195313;
end
    
xs = -1:dx:1;
time = 0:dt:0.005;
T = length(time);
R = (dt)/(dx^2);

u = zeros(1,length(xs)+2);
u(2:end-1) = f(xs);
u(1) = u(2);
u(end) = u(end-1);

error = zeros(1,T);

for t = 1:T
    s = sol(xs,time(t));
    old = u;
    u(1) = u(2);
    u(end) = u(end-1);
    u(2:end-1) = old(2:end-1) + R*(old(3:end)-2*old(2:end-1)+old(1:end-2));
    u(2:end-1) = u(2:end-1) + p*dt*old(2:end-1).*(ones(1,length(xs))-old(2:end-1));
    %Error
    %error(t) = abs(norm(s)-norm(u));
    %Plot
    figure(1);
    plot(xs,u(2:end-1),xs,s);
    axis([-1,1,0,1])
    pause(0.001);
end

hold off
%pause(1);
%loglog(time,error);