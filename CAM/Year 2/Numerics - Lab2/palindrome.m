function p = palindrome()
x = [100:999];
A = unique(transpose(x)*x);
i = length(A);
while strcmp(num2str(A(i)),flip(num2str(A(i))))==0
    i = i-1;
end
p = A(i);
end

