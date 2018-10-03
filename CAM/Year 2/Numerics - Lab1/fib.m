function sum = fib(n)
%
%INPUT: n (scalar)
%OUTPUT: sum of all even-indexed fibonacci numbers <= n
%;
F = [1,1];
i = 3;
sum =0;
while ((F(i-2)+F(i-1))<=n)
    f = F(i-2)+F(i-1);
    F = [F f];
    i = i+1;
end
for j=2:2:length(F)
    sum = sum + F(j);
end
end