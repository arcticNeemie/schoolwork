function mat  = Givens(A)
% B = [1, sqrt(2), sqrt(2), 2; sqrt(2) , -sqrt(2), -1, sqrt(2); 
%     sqrt(2), -1,  sqrt(2), sqrt(2);2, sqrt(2), sqrt(2),-3]
[r,c] = size(A);
%nested (double) for loop because we are dealing with a matrix
%we start at row 1
for i=1:r
    %and column i+2 = 1+2 = 3, because we want a TRI-diagonal matrix
    for j =i+2:c
        %calculation of theta
        %A(i,j) is the position you want to make zero
        %A(i, i+1) is the position of the right-most tridiagonal number in
        %row i
        theta = -atan(A(i,j)/A(i, i+1));
        %Create identity matrix and start filling in cos and sin where
        %appropriate
        P = eye(r);
        P(i+1,i+1) = cos(theta);
        P(j,j) = cos(theta);
        P(i+1,j) = sin(theta);
        P(j,i+1) = -sin(theta);
        %calculate new matrix A
        A = transpose(P)*A*P;
    end
end
mat = A;
end