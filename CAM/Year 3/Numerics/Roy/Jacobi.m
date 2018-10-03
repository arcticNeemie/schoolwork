function eigan = Jacobi(A,it)
% A = [1,3,4;3,2,8;4,8,3]

%function takes in matrix, A and number of iteration, it
[r,c] = size(A);
%for loop will compute new A up until number of iterations
for i=1:it
    %get location of largest number in matrix A
    [m,n] = getMax(A);
    %calculate theta
    theta = 0.5*atan(2*A(m,n)/(A(n,n) - A(m,m)));
    %Create identity matrix and start filling in cos and sin where
    %appropriate
    P = eye(r);
    P(m,m) = cos(theta);
    P(n,n) = cos(theta);
    P(m,n) = sin(theta);
    P(n,m) = -sin(theta);
    %calculate new matrix A
    A = transpose(P)*A*P;
  
end

%saves the diagonal of new matrix A into a vector, eigan 

for j=1:r
    eigan(j) = A(j,j);
end

end


%function takes in matrix A and returns location i,j of the largest
%absolute number in the matrix - that is not in the diagonal
function [i_max, j_max] = getMax(A)
[r,c] = size(A);
%lowest number possible
max = -Inf;
%nested (double) for loop because we are searching through a matrix
for i=1:r
    for j=1:c
        %if statement makes sure we dont look at the diagonal
        if i~=j
            %we save the absolute value of a number in the matrix, call it
            %temp
            temp = abs(A(i,j));
            %then we check whether it is larger than our largest number
            %if it is, then temp because our new max number
            %if not, we skip it
            if temp>max
               max =temp;
               %saving location of the new max number
               i_max = i;
               j_max = j;
            end
        end
    end
end



end
