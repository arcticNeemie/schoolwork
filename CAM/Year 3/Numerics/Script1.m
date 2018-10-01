clc
close all
clear all


f = @(x) sin(4*x);
w = @(x) 1./sqrt(1-x.^2);
xs = @(x) x;


x = -1:0.01:1;
phis = genPhis(1,x);
hold on
plot(x,f(x));
for m = 1:5
    g = zeros(1,length(x));
    for i=0:m
      integrand = @(xs) w(xs).*f(xs).*genPhis(i,xs);
      g = g + integral(integrand,-1,1)*genPhis(i,x);
    end
    plot(x,g);
end
hold off
%Plot a bunch of Chebyshev polynomials
% hold on
% for i = 0:50
%     plot(x,genTns(i,x));
% end
% hold off
%    

% Plot a bunch of Phis
% hold on
% for i=0:20
%     plot(x,genPhis(i,x));
% end
% hold off
