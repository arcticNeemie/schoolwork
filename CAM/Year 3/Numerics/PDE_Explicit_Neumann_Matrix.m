%Neumann, Explicit, Matrix
clc
clear all
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

b0 = -1;
b1 = -1;

%x
x = x0:dx:x1;
N = length(x);
%time
time = t0:dt:t1;
T = length(time);

%Initial Condition
u = transpose(f(x));

%Boundary Condition
% u(1) = b0;
% u(end) = b1;

L1 = (D*dt)/(dx^2);
L2 = 1 - 2*L1;

%Construct B
B = zeros(N,N);
B(1,1) = L2;
B(1,2) = 2*L1;
B(end,end) = L2;
B(end,end-1) = 2*L1;
for i=2:N-1
    B(i,i) = L2;
    B(i,i-1) = L1;
    B(i,i+1) = L1;
end

%Construct b
b = zeros(N,1);
b(1) = 2*L1*dx*b0;
b(end) = 2*L1*dx*b1;

%Iterate across time
for t=1:T
    old = u;
    u = B*old+b;
    
    %Plot
    figure(1);
    plot(x,u);
    axis([x0,x1,0,1])
    pause(0.001);
end

hold off