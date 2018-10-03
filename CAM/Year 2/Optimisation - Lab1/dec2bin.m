function bin = dec2bin(n)
%
%INPUT: n - a number in base 10
%OUTPUT: bin - n in base 2
%
b = char();
while n>0
    i = mod(n,2);
    b = [num2str(i),b];
    n = floor(n/2);
end
bin = char(b);
end