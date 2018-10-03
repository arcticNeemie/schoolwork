function ans = neighbour(v)
n = length(v);
if n>1
    ans = zeros(1,n-1);
    for i=1:n-1
        ans(i) = abs(v(i)-v(i+1));
    end
else
    ans = [];
end
end