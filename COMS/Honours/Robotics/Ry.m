function R = Ry(theta,dx,dy,dz)
     %
    %Input: theta (in degrees), dx, dy, dz - all scalars
    %Output: R, a homogeneous transformation matrix rotated anticlockwise 
    %by theta about y axis and translated by dx,dy,dz in x,y,z
    %
    theta = theta*pi/180;
    R = [cos(theta),0,sin(theta),dx;
        0,1,0,dy;
        -sin(theta),0,cos(theta),dz;
        0,0,0,1]
end