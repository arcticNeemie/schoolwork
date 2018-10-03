function B = makeBoard(n,m)
%
% INPUT: n,m - both scalars
% OUTPUT: B - a nxm matrix consisting of 1's and 0's
%

B = zeros(n,m);
if m~=0 || n~=0
    for i=1:n
        for j=1+mod(i+1,2):2:n
            B(i,j) = 1;
        end
    end
end
end