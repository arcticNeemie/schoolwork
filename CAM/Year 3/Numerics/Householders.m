function A = Householders(A)
    %
    %Inputs
    %   A - a square, symmetric matrix
    %Outputs
    %   A - a tri-diagonal reduction of A
    %
    [r,c] = size(A);
    %Check for Square
    if r~=c
        disp('Error executing Householder method: Matrix not square');
        return;
    end
    %Loop through rows
    for i=1:c-2
       x = transpose(A(i,i+1:c)); %A(i,i+1) to end of row
       alpha = norm(x);
       %Basis vector
       e = zeros(c-i,1);
       e(1) = 1;
       %U
       u = x - alpha*e;
       w = (1/norm(u))*u;
       H = eye(r-i)-2*w*transpose(w);
       I = eye(i);
       Z = zeros(r-i,i);
       P = [I,transpose(Z);Z,H];
       A = P*A*P;
    end
end