function out = heatN(dx,dt,D,it)
x = 0:dx:1;
N = length(x);
f = @(x) x.*(1-x).*(x-0.5);
%f = @(x) exp(-100*(x-0.5).^2);

L1 = (D*dt)/(dx^2);
L2 = 1-2*L1;

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

b0 = -0.5;
b1 = -0.5;

b = zeros(N,1);
b(1) = -2*L1*dx*b0;
b(end) = 2*L1*dx*b1;

u = transpose(f(x));

for i = 1:it
    old = u;
    u = B*old+b;
    
%     figure(1)
%     plot(x,u)
%     axis([0,1,0,1])
%     pause(0.01)
end

out = u;
end 