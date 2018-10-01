function eigs = Jacobi(A,tol)
    %
    %Inputs
    %   A - a square, symmetric matrix
    %   tol - the tolerance
    %Outputs
    %   eigs - a vector containing the eigenvalues of A
    %
    symmetric = checkSymmetry(A);
    if symmetric ~= 1
        disp('Error executing Jacobi Method');
        return;
    end
    [r,~] = size(A);
    oldA = A;
    P = makeP(A);
    A = transpose(P)*A*P;
    while maxDiff(A,oldA)>tol
        P = makeP(A);
        oldA = A;
        A = transpose(P)*A*P;
    end
    eigs = zeros(r,1);
    for i=1:r
        eigs(i)=A(i,i);
    end
end

function max = maxDiff(A,B)
    %Finds the maximum elementwise difference between A and B
    max = 0;
    [r,~] = size(A);
    for i=1:r
        for j=1:r
            curr = abs(A(i,j)-B(i,j));
            if curr>max && i~=j
                max = curr;
            end
        end
    end
end

function P = makeP(A)
    %Creates the P matrix
    n = size(A,1);
    [r,c] = findMaxOffDiagIndices(A); 
    th = 0.5*atan(2*A(r,c)/(A(c,c)-A(r,r)));
    P = eye(n);
    P(r,r) = cos(th);
    P(c,c) = cos(th);
    indices = [r,c];
    P(min(indices),max(indices)) = sin(th);
    P(max(indices),min(indices)) = -sin(th);
end

function [ir,ic] = findMaxOffDiagIndices(A)
    %Returns the row and column of the maximum off-diagonal element in A
    [r,c]=size(A);
    max = -Inf;
    for i=1:r
        for j=1:c
            if abs(A(i,j))>max && i~=j
                ir = i;
                ic = j;
                max = abs(A(i,j));
            end
        end
    end
end

function symmetric = checkSymmetry(A)
    %Checks if A is symmetric. Returns 1 if yes, -1 if no
    symmetric = 1;
    [r,c] = size(A);
    if r~=c
        disp('Matrix is not square');
        symmetric = -1;
    else
        for i=1:r
            for j=1:r
                if A(i,j)~=A(j,i)
                    symmetric = -1;
                    disp('Matrix is not symmetric');
                    return;
                end
            end
        end
    end
end