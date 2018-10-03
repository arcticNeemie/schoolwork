function [stp] = stationFinder(f, g, x)
    stp = [];
    for i=1:length(x)
       if g(x(i)) == 0
           stp = [stp; x(i), f(x(i))];
       end
    end
end