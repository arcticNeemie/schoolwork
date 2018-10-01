function Phin = genPhis(n,x)
    %
    %Input: n, a scalar
    %       x, a vector
    %Output: Phi_n(x)
    %
    if n==0
        Phin = (1/sqrt(pi))*ones(1,length(x));
    else
        Phin = sqrt(2/pi)*genTns(n,x);
    end
end