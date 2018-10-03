function sum = sumMultiplesThreeFive(n)
%
%INPUT: n
%OUTPUT: Sum of all multiples of 3 or 5 less than n
%
sum = 0;
i = 1;
while (3*i < n)
    sum = sum + 3*i;
    i = i+1;
end
i=1;
while ((5*i<n)&&(mod(5*i,3)~=0))
    sum = sum + 5*i;
    i = i+1;
end
end