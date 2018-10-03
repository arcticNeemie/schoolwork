function [c1,c2] = countNegative(A)
    c1 = 0;
    c2 = 0;
    for i=1:size(A,1)
        if A(i,1) < 0 || A(i,2) < 0
            c1 = c1+1;
        end
    end
    C = convexHullGen(A);
    for i=1:size(C,1)
        if C(i,1) < 0 || C(i,2) < 0
            c2 = c2+1;
        end
    end
end

function vertPts = convexHullGen(pts)
    set = [];
    for i=1:size(pts,1)
        if pts(i,1)>=0 && pts(i,2) >=0
            set = [set; pts(i,:)];
        end
    end
    set = unique(set,'rows');
    p1 = set(:,1);
    p2 = set(:,2);
    k = convhull(p1,p2);
    vertPts = [p1(k),p2(k)];
end