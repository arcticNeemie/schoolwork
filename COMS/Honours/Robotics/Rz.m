function R = Rz(theta,dx,dy,dz)
     %
    %Input: theta (in degrees), dx, dy, dz - all scalars
    %Output: R, a homogeneous transformation matrix rotated anticlockwise 
    %by theta about z axis and translated by dx,dy,dz in x,y,z
    %
    theta = theta*pi/180;
    R = [cos(theta),-sin(theta),0,dx;
        sin(theta),cos(theta),0,dy;
        0,0,1,dz;
        0,0,0,1]
end