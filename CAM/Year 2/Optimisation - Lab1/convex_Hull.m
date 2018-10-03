function lst = convex_Hull(points)
%
% INPUT: points - a nx2 matrix of points
% OUTPUT: lst - a mx2 matrix consisting of the points that lie on the convex hull of points
%
n=size(points,1);
points = sortrows(points);

p1=points(1,:); %initial
pn=p1; %last point
i=1;
ep=0; %end point
lst=[];
while ep ~= p1
    lst=[lst; pn];
    ep=points(1,:); %initial ep for candidate edge on the hull
    for j=2:n
        d=(points(j,1)-lst(i,1))*(ep(1,2)-lst(i,2))-(points(j,2)-lst(i,2))*(ep(1,1)-lst(i,1));
        %cross product used to determine if point is on left side of line
        if isequal(ep,pn) || (d<0) % hull point or left of line
            ep=points(j,:);
        end
    end
    %line([pn(1) ep(1)],[pn(2) ep(2)])
    %to display convex hull
    i=i+1;
    pn=ep;
end
end