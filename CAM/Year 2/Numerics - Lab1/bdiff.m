function df = bdiff(f,x,h)
%
%INPUT: function f
%OUTPUT: returns backward difference approximation of f'
%
df = (f(x)-f(x-h))/h;
end


