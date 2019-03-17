function A = DH(a,alpha,d,theta)
     %
    %Input: a, alpha, d, theta - all scalars (angles in degrees)
    %Output: A, the corresponding Denavit-Hartenberg transformation matrix
    %
    theta = theta*pi/180;
    alpha = alpha*pi/180;
    A = [cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta);
        sin(theta),cos(theta)*cos(alpha),-cos(theta)*sin(alpha),a*sin(theta);
        0,sin(alpha),cos(alpha),d;
        0,0,0,1];
end