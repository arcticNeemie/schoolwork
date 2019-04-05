syms a alpha d theta
A = [cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta);
    sin(theta),cos(theta)*cos(alpha),-cos(theta)*sin(alpha),a*sin(theta);
    0,sin(alpha),cos(alpha),d;
    0,0,0,1];

syms theta1
a = 0;
alpha = 0;
d = 0;
theta = theta1;
T10 = subs(A);

syms d2
alpha = -pi/2;
d = d2;
theta = 0;
T21 = subs(A);

syms d3
alpha = 0;
d = d3;
T32 = subs(A);

T30 = T10*T21*T32;
T20 = T10*T21;
