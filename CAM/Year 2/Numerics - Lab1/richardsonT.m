function rTab = richardsonT(f, x, h)
%
%INPUT: (f,x,h)
%   f is a function
%   x is the point on the domain of f
%   h is a vector of step sizes
%OUTPUT: a richardson extrapolation table as a matrix
%
rTab = NaN(size(h,1),size(h,1)); 
for i=1:size(h,1)
    for j=1:i
        if(j==1)
            rTab(i,1)=cdiff(f,x,h(i));
        else
            rTab(i,j)=(1/((4^(j-1))-1))*(4^(j-1)*rTab(i,j-1)-rTab(i-1,j-1));
        end
    end
end
end

%m = rows
%i = cols -1

function df = cdiff(f,x,h)
%
%INPUT: function f
%OUTPUT: returns central difference approximation of f'
%
df = (f(x+h)-f(x-h))/(2*h);
end

function integral = trap(f,a,b,h)
n = (b-a)/h;
%if Chad gives us n, h = (b-a)/n
integral = f(a);
for i=1:n-1
    integral = integral + 2*f(a+i*h);
end
integral = (h/2)*(integral + f(b));
end

function integral = simpson(f,a,b,h)
n = (b-a)/h;
F = ones(1,n+1);
F(1) = f(a);
for i=1:n
    F(i+1) = f(a + i*h);
end
sum = 0;
for i=1:2:n-1
    sum = sum + F(i)+4*F(i+1)+F(i+2);
end
integral = (h/3)*sum;
end
