function min = steepDesAlpha(f, g, x0, alpha, iter)
%f = @(x,y) 2*x.^2+3*y.^2
%g = @(x,y) [4*x 6*y]
%x0 =[1 1]
%alpha = 13/70
%I could not check this fully, but the first iteration (iter=1) corresponds
%to the notes answers and a larger number of iterations tends to zero for
%this example so I hope it works

% This is the answer to the question given in the lab, not moodle

n = length(x0);
%I am making a cell array, the reason for this is that I could not find
%another way to send a vector into the unknown MC function, if you find a better
%way, please tell me
arr = cell(1, n);
%We send the vector into the cell array
%To be clear, arr is exactly the same as x0, but it can be used differently
for j=1:n
        arr{j} = x0(j);
end

for i=1:iter
    %This is the formula for the next vector x, you can see that if try:
    %x0 = x0-alpha*(g(x0)); - it will not work, this is why I made the cell
    %array
    x0 = x0-alpha*(g(arr{:}));
    %now i just send the new vector into the cell array again
    for j=1:n
        arr{j} = x0(j);
    end
end
min = f(arr{:});
end