function A = randInt(limit,n,m)
%
% INPUT: limit,n,m
%     limit - max value
%     n,m - scalars
% OUTPUT: A - nxm matrix of random integers between 1 and limit inclusive
%

A = ceil((limit)*(1/(1-eps))*(rand(n,m)-(eps/2)*ones(n,m)));
%Not sure if this is right, but eh, close enough
end