  experiment = 4;
  err = zeros(1, experiment);
  
  bc = 'D';
  %bc = 'N';
  isSimulate = false;
  isPlot3D = false;
  
  
  for i = 1:experiment
    fprintf('Experiment: %i\n', i)
    switch i
      case 1
            dx = 0.05;
            dt = 0.00125;
      case 2
            dx = 0.025;
            dt = 0.0003125;
      case 3
            dx = 0.0125;
            dt = 0.000078125;
      case 4
            dx = 0.00625;
            dt = 0.0000195313;
    end
    
    if bc == 'D'
      [curr_error] = NSFD_Dirichlet(dx, dt, i, isSimulate, isPlot3D);
    else
      [curr_error] = NSFD_Neumann(dx, dt, i, isSimulate, isPlot3D)
    end
    err(i) = curr_error
    
  end
  
  % Plot Error
  figure(3)
  plot([1:4], err)
  set(gca, 'yscale', 'log')
  xlabel('Experiment')
  ylabel('Error')
  