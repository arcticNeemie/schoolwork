function [err] = NSFD_Dirichlet(dx, dt, experiment, isSimulate, isPlot3D)

  p = 2000;
  f = @(x) 1./((exp(10*sqrt(10/3)*x)+1).^2);
  sol = @(x,t) 1./((exp(sqrt(p/6).*x-(5*p/6)*t)+1).^2);
  
  xs = -1:dx:1;
  time = 0:dt:0.005;
  T = length(time);
  p = 2000;
  R = (dt)/(dx^2);
  
  un = f(xs);
  un(1) = 1;
  un(end) = 0;
  u_points = zeros(length(T), length(xs));
  
  error_exp = zeros(1, T);
  err = 0;
  
  
  for n = 1:T
    
    old = un;
    C = (1+p*dt-2*R);
    un(2:end-1) = (R.*(old(3:end)+old(1:end-2)) + C.*old(2:end-1))./(ones(1,length(xs)-2)+((p*dt)/3).*(old(3:end)+old(2:end-1)+old(1:end-2)));
    u_points(n, :) = un;
    
    % true solution
    u_sol = sol(xs,time(n));
    
    error_exp(n) = sum((un-u_sol).^2);
    
    if(n == T)
      err = error_exp(n);
    end
    
    % simulation
    if isSimulate
      figure(1)
      plot(xs, un, xs, u_sol);
      legend({'NSFD solution','True solution'},'Location','northeast')
      title(['Experiment ', num2str(experiment)])
      axis([-1 1 0 1])
      pause(0.01);
    end
    
  end
   
   % 3D plot
   if isPlot3D
     figure(2)
     mesh(xs, time, u_points)
     xlabel('x')
     ylabel('t')
     zlabel('u(x,t)')
   end
   
%  figure(4)
%  plot([1:T], error_exp)
  
end

