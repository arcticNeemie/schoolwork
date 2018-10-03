
function [Q, R] = QRMethod(A)
%A =[4 1 1; 1 4 1; 1 1 4]
[r,c] = size(A);
Q = eye(r,c);
%loop until the  last row, this time starting out i=0
for i = 0:c-1
    
    %creates vector x, starting from the right of diagonal this time, 
    %to the end of the row
    %Since we start at zero, to select the row this time we must have 
    %A(i+1, ...)
    x = transpose(A(i+1,i+1:c));
    
    %constructs the unit vector that we need for this method
    unit = zeros(c-i,1);
    unit(1) = 1;
    
    alpha = norm(x);
        if x(1)>0
            alpha = -alpha;
        end
    u = x - alpha*unit;
    w = u/norm(u);
    
    %note the identity matrix changes each iteration - it gets smaller
    %each time eye(r-i). But for the first iteration, it will have the same
    %dimensions as A
    H = eye(r-i) - 2*w*transpose(w);
    
    %we make same matrix as for householder method, however, since we
    %started at i=0, lc,lb and rc will be empty matrices in the first
    %iteration. This implies for the 1st iteration P=H - which is correct.
    
    lc = eye(i);
    rc = zeros(i, c-i);
    lb = zeros(c-i, i);
    top = [lc, rc];
    bot = [lb, H];
    P = [top; bot];
    
    %get new matrix A
    A = P*A;
    %Q is just the combinations of your P matrices i.e. Q = P_1 * P_2 * P_3
    %...
    Q = Q*P;
end
R = A;
end