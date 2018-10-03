function P = randomSample(x,r,n,s)
%
% INPUT: (x,r,n,s)
%     x - a vector in R2 around which points will be sampled
%     r - a radial scalar
%     n - the umber of points to be generated
%     s - the shape of the neighbourhood. s=1 for a circle, s=2 for a square
% OUTPUT: P - an n*2 matrix containing the points
%

if s==1
    P = [];
    for i=1:n
        area=2*pi*rand;
        rad=sqrt(rand);
        xi=(r*rad)*cos(area)+x(1);
        yi=(rad*r)*sin(area)+x(2);
        P = [P;xi,yi];
    end
elseif s==2
    xn = [];
    for i=1:n
        xn = [xn;x(1),x(2)];
    end
    Pr = 2*r*rand(n,2)-r*ones(n,2);
    P = xn + Pr;
end
end

%A = randomSample(x,r,n,s);
%a1 = A(:,1);
%a2 = A(:,2);
%plot(a1,a2,'o')
