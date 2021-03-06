%Finite Difference Heat Equation - Matrix Implementation (Neumann)
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

b0 = -1;
b1 = -1;

%x
xs = x0:dx:x1;
N = length(xs);
%time
time = t0:dt:t1;
T = length(time);

u = transpose(f(xs));
L1 = (D*dt)/(dx^2);
L2 = 1 - 2*L1;
alpha = -1;
beta = -1;

B = zeros(N,N);
b = zeros(N,1);
b(1) = 2*L1*dx*alpha;
b(N) = 2*L1*dx*beta;
B(1,1) = L2;
B(1,2) = 2*L1;
B(N,N) = L2;
B(N,N-1) = 2*L1;
for i=2:N-1
    B(i,i) = L2;
    B(i,i-1) = L1;
    B(i,i+1) = L1;
end

for n = 1:T
    old = u;
    u = B*old+b;
    %Plot
    figure(1);
    plot(xs,u);
    axis([0,1,0,1])
    pause(0.001);
end

hold off