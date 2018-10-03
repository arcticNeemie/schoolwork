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