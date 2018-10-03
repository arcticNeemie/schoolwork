function outM = modExtraction(M)
outM = [];
for i=2:2:size(M,1)
    x = [];
    for j=2:2:size(M,2)
        x = [x M(i,j)];
    end
    outM = [outM;x];
end

end