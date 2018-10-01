%Finite Difference Heat Equation - Matrix Implementation (Neumann)
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