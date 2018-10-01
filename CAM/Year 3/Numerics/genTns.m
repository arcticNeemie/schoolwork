function Tn = genTns(n,x)
    %
    %Input: n, a scalar
    %       x, a vector
    %Output: Tn, the n'th order Chebyshev polynomial, denoted T_n(x)
    %
    if n==0
        Tn = ones(1,length(x));
    elseif n==1
        Tn = x;
    else
        Tn = 2*x.*genTns(n-1,x) - genTns(n-2,x);
    end
end