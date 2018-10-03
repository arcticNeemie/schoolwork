function v = order3(x,y,z)
if x<y && y<z
    min = x;
    med = y;
    max = z;
elseif x<z && z<y
    min = x;
    med = z;
    max = y;
    elseif y<z && z<x
    min = y;
    med = z;
    max = x;
    elseif y<x && x<z
    min = y;
    med = x;
    max = z;
    elseif z<x && x<y
    min = z;
    med = x;
    max = y;
    elseif z<y && y<x
    min = z;
    med = y;
    max = x;
end
v = [min,med,max];
end