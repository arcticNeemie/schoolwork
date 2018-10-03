function oddM = odds(M)
%
%INPUT: M - a matrix
%OUTPUT: oddM - a matrix constructed from the odd rows and columns of M
%
oddM = [];
for i=1:2:size(M,1)
    x = [];
    for j=1:2:size(M,2)
        x = [x M(i,j)]
    end
    oddM = [oddM;x];
end

end