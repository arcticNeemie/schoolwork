function [lst] = vecCompute(x)
xSort = sort(x);
lst = ones(1,3);
lst(1) = xSort(1);
if mod(size(xSort,2),2)== 0
    lst(2) = 0.5*(xSort((size(xSort,2)/2)+1)+xSort(size(xSort,2)/2));
else
    lst(2) = xSort(ceil(size(xSort,2)/2));
end
lst(3) = xSort(length(xSort));
end