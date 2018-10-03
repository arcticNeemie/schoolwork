function mat = Householder(A)
%C = [3,2,1,2;2,-1,1,2;1,1,4,3;2,2,3,1]
[r,c] = size(A);

%loop until the 2nd last row
for i = 1:c-2
    %creates vector x, starting from number right of diagonal, to the end
    %of the row
    x = transpose(A(i,i+1:c));
    %constructs the unit vector that we need for this method
    unit = zeros(c-i,1);
    unit(1) = 1;
    %gets alpha by finding the norm of x
    alpha = norm(x);
    %ensures that the first component of x has an opposite sign to alpha
        if x(1)>0
            alpha = -alpha;
        end
    %from the notes
    u = x - alpha*unit;
    w = u/norm(u);
    %note the identity matrix changes each iteration - it gets smaller
    %each time eye(r-i)
    H = eye(r-i) - 2*w*transpose(w);
    %okay, we construct the matrix P by splitting it into 4 chunks. We know
    %what its composed of - see notes. The bottom right hand corner is the
    %matrix H we just calculate. Now:
    
    % * left corner - lc - is an identity matrix that gets larger each time we
    %iterate, thus eye(i)
    lc = eye(i);
    % * right corner - rc - is a rectangle of zeros 
    rc = zeros(i, c-i);
    % * left bottom - lb - is also a rectangle of zeros, but rotated
    % compared to rc. Note that we can also write lb = transpose(rc)
    lb = zeros(c-i, i);
    %glue the top parts of the matrix
    top = [lc, rc];
    %glue the bottom parts
    bot = [lb, H];
    %Combine it all
    P = [top; bot];
    %get new matrix A
    A = P*A*P;
end
mat = A;
end