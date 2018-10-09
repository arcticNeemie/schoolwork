function out = wave(dx,dt,omega,it)
  x = 0:dx:1;
  f = @(x) exp(-100*(x-0.5).^2);
  
  R = ((omega^2)*(dt^2))/(dx^2);
  
  u0 = f(x);
  u0(1) = 0;
  u0(end) = 0;
  u1 = u0;
  
  for i = 1:it
      u2 = u1;
      u2(2:end-1) = 2*u1(2:end-1) + R*(u1(3:end)-2*u1(2:end-1)+u1(1:end-2)) - u0(2:end-1);
      u0 = u1;
      u1 = u2;

%     figure(1)
%     plot(x,u1)
%     axis([0,1,-1,1])
%     pause(0.001)
  end
  
  out = u1;
end