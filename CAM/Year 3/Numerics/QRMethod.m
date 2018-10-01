function [Q,R] = QRMethod(A)
	%
    %Inputs
    %   A - a square matrix
    %Outputs
    %   Q, R - matrices such that A = QR, with Q orthonormal and R upper
    %   triangular
    %
    [r,c] = size(A);
    %Check for Square
    if r~=c
        disp('Error executing QR method: Matrix not square');
        return;
    end
    Q = eye(r);
    for i=1:c
       x = transpose(A(i,i:c)); %A(i,i) to end of row
       alpha = norm(x);
       if x(1)>0
        alpha = -alpha;
       end
       e = zeros(c-i+1,1);
       e(1) = 1;
       u = x - alpha*e;
       w = (1/norm(u))*u;
       H = eye(r-i+1)-2*w*transpose(w);
       I = eye(i-1);
       Z = zeros(r-i+1,i-1);
       P = [I,transpose(Z);Z,H];
       A = P*A;
       Q = Q*P;
    end
    R = A;
end