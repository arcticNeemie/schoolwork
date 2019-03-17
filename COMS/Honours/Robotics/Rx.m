function R = Rx(theta,dx,dy,dz)
     %
    %Input: theta (in degrees), dx, dy, dz - all scalars
    %Output: R, a homogeneous transformation matrix rotated anticlockwise 
    %by theta about x axis and translated by dx,dy,dz in x,y,z
    %
    theta = theta*pi/180;
    R = [1,0,0,dx;
        0,cos(theta),-sin(theta),dy;
        0,sin(theta),cos(theta),dz;
        0,0,0,1]
end