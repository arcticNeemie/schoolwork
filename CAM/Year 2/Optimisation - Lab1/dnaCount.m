function c = dnaCount(s)
%
%INPUT: s - a vector of strings A,C,G or T
%OUTPUT: a 1x4 vector containing the number of times A,C,G and T occur in
%the string respectively
%

c = zeros(1,4);
for i=1:length(s)
    if s(i) == 'A'
        c(1) = c(1) + 1;
    elseif s(i) == 'C'
        c(2) = c(2) + 1;
    elseif s(i) == 'G'
        c(3) = c(3) + 1;
    elseif s(i) == 'T'
        c(4) = c(4) + 1;
    end
end
end