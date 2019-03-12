syms alpha1 alpha2 d1 d2 d3 th1 th2

T10_th = [cos(th1),-sin(th1),0,0;sin(th1),cos(th1),0,0;0,0,1,0;0,0,0,1];
T10_d1 = [1,0,0,0;0,1,0,0;0,0,1,d1;0,0,0,1];
T10_alpha = [1,0,0,0;0,cos(alpha1),-sin(alpha1),0;0,sin(alpha1),cos(alpha1),0;0,0,0,1];
T10 = T10_th*T10_d1*T10_alpha;

T21_th = [cos(th2),-sin(th2),0,0;sin(th2),cos(th2),0,0;0,0,1,d2;0,0,0,1];
T21_alpha = [1,0,0,0;0,cos(alpha2),-sin(alpha2),0;0,sin(alpha2),cos(alpha2),0;0,0,0,1];
T21 = T21_th*T21_alpha;
T20 = T10*T21;

T32 = [1,0,0,0;0,1,0,0;0,0,d3,0;0,0,0,1];
T30 = T10*T21*T32;

alpha1 = pi/2;
alpha2 = -pi/2;
subs(T30)