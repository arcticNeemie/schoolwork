function Q = quadrant(n)
Q4 = ones(n,n);
Q3 = 2*Q4;
Q2 = 3*Q4;
Q1 = 4*Q4;
Q = [Q1,Q2;Q3,Q4];
end