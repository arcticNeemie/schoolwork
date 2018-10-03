function p = sieve(n)
    ns = [2:n];
    for i=2:n
        for j=i+i:i:n
            ns(j-1) = 2;
        end
    end
    p = unique(ns);
end