function cash = countMoney(c)
%
%INPUT: c - a 1x4 vector where:
%     c1 = number of 5 cents
%     c2 = number of 10 cents
%     c3 = number of 20 cents
%     c4 = number of 50 cents
%OUTPUT: cash - the total amount of $$$ we have in rands
%
cash = c(1)*0.05 + c(2)*0.1 + c(3)*0.2 + c(4)*0.5;
end