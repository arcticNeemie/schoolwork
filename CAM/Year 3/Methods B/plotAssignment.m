%Inputs
a = 1000;
b = 500;
alpha = 0.1;
m = 10;

%Domains
x = 0:0.1:a;
z = 0:0.1:b;
X = length(x);
Z = length(z);

U = zeros(X,Z);
for i=1:X
    for j=1:Z
        u = b + alpha*a/2;
        for n=1:m
            u = u + ((alpha*(a^2)*cos(n*pi)-1))/((n^2)*(pi^2)*cosh(n*pi*b/a));
            u = u*cos(n*pi*x(i)/a)*cosh(n*pi*z(j)/a);
        end
        U(i,j) = u;
        disp(i+','+j);
    end
end
U