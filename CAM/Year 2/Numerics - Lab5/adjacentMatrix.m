function s = adjacentMatrix(x) 
r = size(x,1);
c = size(x,2);
s = 0;
%vertical
if r>=4
    for i=1:r-3
        for j=1:c
            p = x(i,j)*x(i+1,j)*x(i+2,j)*x(i+3,j);
            if p>s
                s = p;
            end
        end
    end
end
%horizontal
if c>=4
    for j = 1:c-3
        for i=1:r
            p = x(i,j)*x(i,j+1)*x(i,j+2)*x(i,j+3);
            if p>s
                s = p;
            end
        end
    end
end
%diagonal down-right
if c>=4 && r>=4
    for i=1:r-3
        for j=1:c-3
            p = x(i,j)*x(i+1,j+1)*x(i+2,j+2)*x(i+3,j+3);
            if p>s
                s = p;
            end
        end
    end
end
%diagonal up-right
if c>=4 && r>=4
    for i=r:-1:4
        for j=1:r-3
            p = x(i,j)*x(i-1,j+1)*x(i-2,j+2)*x(i-3,j+3);
            if p>s
                s = p;
            end
        end
    end
end
end