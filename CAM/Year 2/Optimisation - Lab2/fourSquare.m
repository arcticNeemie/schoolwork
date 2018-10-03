function Q = fourSquare(n)
%
%INPUT: n - a scalar
%OUTPUT: Q - a 2nx2n matrix of 1's,2's,3's and 4's
%
Q1 = ones(n,n);
Q2 = 2*Q1;
Q3 = 3*Q1;
Q4 = 4*Q1;
Q = [Q1,Q2;Q3,Q4];
end