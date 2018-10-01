function A = Givens(A)
    %
    %Inputs
    %   A - a square, symmetric matrix
    %Outputs
    %   A - a tri-diagonal reduction of A
    %
    symmetric = checkSymmetry(A);
    if symmetric ~= 1
        disp('Error executing Givens Method');
        return;
    end
    [r,~] = size(A);
    for i=1:r
        for j = i+2:r
            P = makeP(A,i,j);
            A = transpose(P)*A*P;
        end
    end
end

function symmetric = checkSymmetry(A)
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

function P = makeP(A,i,j)
    n = size(A,1);
    th = -1*atan(A(i,j)/A(i,i+1));
    P = eye(n);
    P(i+1,i+1) = cos(th);
    P(j,j) = cos(th);
    P(i+1,j) = sin(th);
    P(j,i+1) = -sin(th);
end