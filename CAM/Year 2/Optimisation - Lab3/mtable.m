function [re s] = mtable(n,m)
s = 0;
re = zeros(n,m);
for i=1:n
    for j=1:m
        re(i,j) = i*j;
        s = s + i*j;
    end
end
end