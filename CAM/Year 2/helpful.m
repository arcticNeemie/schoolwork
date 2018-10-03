%%HELPFULL MATLAB FUNCTIONS%%

A = [1,2;3,4];
eig(A); %eigenvalues of A
[V,D] = eig(A); %V is the eigenvectors, D is the eigenvalues

f =@(x) x^2 + 1;
g = @(x,y) x^y;

h = 2;
n = 20;
odds  = []; %empty matrix
for i=1:h:n
    odds = [odds i]; %concatenate i to odds
end
%now odds is the odd numbers from 1 to <20 (19)

x = [1,2,3,4,5]; %row vector
xt = transpose(x); %columnvector

%multiples of 3
%method1
threes = [];
for i=3:n
    threes = [threes i];
end
%method 2
threes = [];
for i=1:n
    if mod(i,3)==0
        threes = [threes i];
    end
end

A = [1,2,3;4,5,6;7,8,9;10,11,12];
r = size(A,1); %rows
c = size(A,2); %cols

x = [1,2,3,4];
l = length(x);

%LENGTH IS FOR VECTORS, SIZE IS FOR MATRICES

string = '12345';
num = str2num(string);

num2 = 12345;
string2 = num2str(num2);

%finding maximum
x = [1,6,8,5,100,54,7,3];
max = x(1);
for i=2:length(x)
    if x(i)>max %< for min
        max = x(i);
    end
end

%indexing
a = 1:10; %creates [1,2,3,4,5,6,7,8,9,10]
A = [a;a;a;a;a;a;a;a;a;a]; % creates a matrix with a as rows, 10 times
i = 3;
j= 4; %for example
p = A(i,j); %matrix at row 3, column 4
p1 = A(i-1,j); %value directly above p
p2 = A(i,j-1); %value directly to the left of p
p3 = A(i+1,j+1); %value to the right and down of p

%n choose r
n = 10;
r = 4;
nCr = factorial(n)/(factorial(r)*factorial(n-r));

%sum of all primes below n
n = 2000000;
p = primes(n); % creates [2,3,5,7,...,prime just before n]
s = sum(p); %adds together all the elements of p e.g. 2+3+5+7+...

%plotting
f = @(x) sin(x);
x = 1:100;
y = f(x);
plot(x,y,'.'); %points
hold on %keeps the same plot
plot(x,sin(x),'-'); %continuous. We get a better plot with smaller intervals of x e.g. x = 1:0.001:100
hold off %so we don't use the same plot

%check if a vector is a column vector
x = 1:100;
if iscolumn(x)
    %x is column
else
    %x is not column
end

%alternatively
if ~iscolumn(x)
    %x is not column
else
    %x is column
end
