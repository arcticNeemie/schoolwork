function abc = pythagTriple(n)
%
% INPUT: n - a scalar
% OUTPUT: abc - the product of a,b and c such that a + b + c = n and a^2 + b^2 = c^2
%
abc = NaN;
for a=1:n/3
    for b=a+1:n/2
        c = n-a-b;
        if a^2 + b^2 == c^2
            abc = a*b*c;
            break;
        end
    end
end

end