function R = Rot2D(theta,dx,dy)
     %
    %Input: theta (in degrees), dx, dy - all scalars
    %Output: R, a homogeneous transformation matrix rotated anticlockwise 
    %by theta and translated by dx in x and dy in y
    %
    theta = theta*pi/180;
    R = [cos(theta),-sin(theta),dx;sin(theta),cos(theta),dy;0,0,1];
end