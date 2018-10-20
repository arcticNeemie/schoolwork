%Inputs
a = 1000;
b = 500;
alpha = 0.1;
m = 10;

%Domains
x = 0:1:a;
z = 0:1:b;
X = length(x);
Z = length(z);

U = zeros(Z,X);
for i=1:X
    for j=1:Z
        u = b + alpha*a/2;
        for n=1:m
            u = u + (2*alpha*a*(cos(n*pi)-1))/((n^2)*(pi^2)*cosh(n*pi*b/a));
            u = u*cos(n*pi*x(i)/a)*cosh(n*pi*z(j)/a);
        end
        U(j,i) = u;
    end
end
[x,z] = meshgrid(x,z);
hold on
surf(x,z,U);
axis([0 1000 0 500 -100 100])
xlabel('x');
ylabel('z');
zlabel('u(x,z)');
hold off
shading interp