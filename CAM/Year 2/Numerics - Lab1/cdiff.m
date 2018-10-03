function df = cdiff(f,x,h)
%
%INPUT: function f
%OUTPUT: returns central difference approximation of f'
%
df = (f(x+h)-f(x-h))/(2*h);
end
