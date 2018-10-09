function out = heatD(dx,dt,D,it)

x = 0:dx:1;
N = length(x);
f = @(x) exp(-100*(x-0.5).^2);

L1 = (D*dt)/(dx^2);
L3 = 1+2*L1;

C = zeros(N,N);
C(1,1) = 1;
C(N,N) = 1;
for i=2:N-1
    C(i,i) = L3;
    C(i,i-1) = -L1;
    C(i,i+1) = -L1;
end

u = transpose(f(x));
u(1) = 0;
u(end) = 0;

for i = 1:it
    u = C\u;
    u(1) = 0;
	u(end) = 0;
    
%     figure(1)
%     plot(x,u)
%     axis([0,1,0,1])
%     pause(0.001)
    
end

out = u;

end