function [coeff,myMatrix] = NewtonDivided(x,y)
%
%INPUT: x,y - vectors of length n
%OUTPUT: coeff - vector of length n
%        myMatrix - n x n matrix  
%
n = length(x);
myMatrix = zeros(n,n-1);
coeff = zeros(n,1);
 %myMatrix
%transpose y
if ~iscolumn(y)
    y = transpose(y);
end
%first column
myMatrix = [y myMatrix];
coeff(1) = y(1);
for j=2:n
    for i=j:n
        myMatrix(i,j) = (myMatrix(i,j-1)-myMatrix(i-1,j-1))/(x(i)-x(i-j+1));
    end
    coeff(j) = myMatrix(j,j); %returns coefficients of (x-x0), (x-x0)(x-x1), etc.
end

end