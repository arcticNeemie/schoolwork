function df = fdiff(f,x,h)
%
%INPUT: function f
%OUTPUT: returns forward difference approximation of f'
%
df = (f(x+h)-f(x))/h;
end