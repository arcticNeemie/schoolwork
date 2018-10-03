%%%SCRIPT%%%
function H = hubble()
w = [3997,3994,3989,3987,3973,3971,4004,3984,3983,4115,3981,3962,4006,3975,3977,3980,3980,3982,3991,3982,4039,3971,4002,3985,3986,4033];
d = [28,25.3,33.72,17.02,9.4,3.5,13.55,20.24,22.99,16.86,9.5,30.05,36.79,10.73,11.04,17.02,17.17,9.2,22.7,9.2,76.65,13.8,35.48,0.01,23.91,115];
n = length(w);
v = galaxyVelocity(w);
if ~iscolumn(d)
    d = transpose(d);
end
X = [ones(n,1) d];
b = X\v; %regression stuff
y = X*b;
plot(d,v,'.');
hold on;
plot(d,y,'-');
%I could try to remove outliers, but meh
H = b(2);
hold off;
end

function v = galaxyVelocity(w)
n = length(w);
v = zeros(n,1);
l0 = 3968;
cm = 299792458; %approximations are for pussies
ckm = cm/1000;
for i=1:n
    v(i) = ckm*(w(i)-l0)/l0;
end
end