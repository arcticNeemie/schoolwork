%Finite Difference Heat Equation - Scalar Implementation
clc
close all


f = @(x) 1./exp((10*(sqrt(10)/3).*x+1).^2);

%Experiment 1
%dx = 0.05;
%dt = 0.00125;

%Experiment 2
%dx = 0.025;
%dt = 0.0003125;

%Experiment 3
%dx = 0.0125;
%dt = 0.000078125;

%Experiment 4
dx = 0.00625;
dt = 0.0000195313;

xs = 0:dx:1;
plot(xs,f(xs));

time = 0:dt:0.005;
T = length(time);
p = 2000;
R = (dt)/(dx^2);

u = f(xs);
u(1) = 1;
u(end) = 0;

for n = 1:T
    old = u;
    u(2:end-1) = old(2:end-1) + R*(old(3:end)-2*old(2:end-1)+old(1:end-2));
    u(2:end-1) = u(2:end-1) + p*dt*old(2:end-1).*(ones(1,length(xs)-2)-old(2:end-1));
    %Plot
    figure(1);
    plot(xs,u);
    axis([0,1,0,1])
    pause(0.001);
end

hold off

