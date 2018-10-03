function s = strAdjacent(x, n)
l = length(x);
s = 0;
for i=1:l-n+1
    p = 1;
    for j=0:n-1
        p = p*str2num(x(i+j));
    end
    if p>s
        s = p;
    end
end
end