function tot = upperSum(A)
r = size(A,1);
c = size(A,2);
tot = 0;
for i=1:r
    for j=i:c
        tot = tot + A(i,j);
    end
end
end