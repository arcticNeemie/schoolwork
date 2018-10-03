function [min,fmin,itr] = DownhillSimplex(xt)

%%Note: I made this programme in such a way as to allow it to apply the
%%Downhill Simplex Method to functions (with corresponding points), in any
%%dimension (i.e. not only the 2 dimensions used in our specific example),
%%so long as the function in RunF is changed to that of which the user
%%wants to apply the method to (I didn't include this as a user input
%%because to do so would mean inputing it as a function handle and this
%%only complicate the process for the marker)



%%ln denotes the number of points we've been given (this is needed since I
%%wish to create the general case of this process instead of the specific 2
%%dimensional case needed for this example

ln=length(xt);
   
   %%First we start with our initial ordering of our points so that we can
   %%atleast start the while loop. To do so, i create a matrix that
   %%contains the points as well as their respective function values
       for i=1:ln
        fx(i,:)=[xt(i,:),RunF(xt(i,:))];
       end
    %%Now i start ordering the points in decending order(worst to best) 
    %%by calling the function OrderF and then i use OrderX to isolate the 
    %%ordered points as rows of the matrix xnew
   fnew = OrderF(fx,ln);
   xnew = OrderX(fnew,ln);
   %%k denotes the number of iterations of the downhill simplex method
   %%performed until our minimum is found or the stopping criteria are
   %%reached 
   k=0;
   %%Here we introduce the while loop so as to ensure the boundaries/
   %%stopping criteria haven't been reached yet
   while (StopCriteria(xnew,ln)>=10^(-50))&&(k<200)
    %%Here we reorder the points again so that every itteration will always
    %%use the points in their correct order
    for z =1:ln
        fx(z,:)=[xnew(z,:),RunF(xnew(z,:))];
    end
    fnew= OrderF(fx,ln);
    xnew=OrderX(fnew,ln);
    
   %%Now we generate a trial point xr by reflection, using our function
   %%Reflection
   xbar=Reflection(xnew,ln);
   
   %%Now we look at the three possibilities
   if (RunF(xnew(ln,:))<=RunF(xbar))&&(RunF(xbar)<=RunF(xnew(2,:)))
      xnew(1,:)=xbar; 
      xbar=Reflection(xnew,ln);
   end
   
   if RunF(xbar)<RunF(xnew(ln,:))
    %%Now we expand the simplex
    xbar2=Expand(xbar,xnew,2,ln);
        if RunF(xbar2)<RunF(xnew(ln,:))
            xnew(1,:)=xbar2;
        elseif RunF(xbar2)>=RunF(xnew(ln,:))
       %%The below operation is one that takes our worst vertex and replace
       %%it with x' bar
               xnew(1,:)=xbar; 
             
        end
       
   elseif RunF(xbar)>RunF(xnew(2,:))
      
       %%Now we contract the simplex using the contraction factor
      if RunF(xbar)<RunF(xnew(1,:))
          %%Note we use the same function  as we did for the expansion, the
          %%only difference is that we used the constant 0.5 instead of 2
          xbar2=Expand(xbar,xnew,0.5,ln);
      elseif RunF(xbar)>=RunF(xnew(1,:))
         
           %%Once again we use the Expansion function but with our worst
           %%vertex instead of x' and a constant of 0.5
            xbar2=Expand(xnew(1,:),xnew,0.5,ln); 
             
         
      end
      if (RunF(xbar2)<RunF(xnew(1,:)))&&(RunF(xbar2)<RunF(xbar))
       %%The below operation is one that takes our worst vertex and replace
       %%it with x'' bar
         xnew(1,:)=xbar2;
         
      end
      if (RunF(xbar2)>=RunF(xnew(1,:)))&&(RunF(xbar2)>RunF(xbar))
          %%Below we see the process of halving the distances from our
          %%'best' vertex (i.e. the vertex with the lowest function value)
          %%to the other vertecies and by doing so, reducing the size of
          %%our simplex
          m1= (xnew(2,:)+xnew(ln,:))/2;
          m2= (xnew(1,:)+xnew(ln,:))/2;
          xnew(1,:)=m2;
          xnew(2,:)=m1;
          
      end
   end
   %%The below operation simply keeps track of the amount of itterations
   %%performed
   k=k+1;
   end
   %%Finally we are able to output our results 
   min=xnew(ln,:);
   fmin=RunF(xnew(ln,:));
   itr=k;
end
function f = RunF(xt)

    %%This function runs a specific point through the provided function,
    %%reading reading the coordinates from a row vector input
    f= (1-xt(1))^(2)+10*(xt(2)-(xt(1)^2))^2;

end
function Order = OrderF(fx,ln)
%%This function reads each point's (except for the last point) 
%%function value and uses it to compare it to all the points in the rows
%%below it

    %%If a function point of a lower row is greater than the row we
    %%are currently analyzing, we swap them. By doing this to the
    %%first n-1 rows (there are n rows in total), we end up with a
    %%set/ matrix of points in decending order of function value
     for j=1:(ln-1)
        for k=1:(ln-j)

          if fx(j,ln)<fx(j+k,ln)
              r1=fx(j,:);
              r2=fx(j+k,:);
              fx(j,:)=r2;
              fx(j+k,:)=r1;
          end
        end
     end
  %%Thus we are able to output an Ordered matrix of the points along with 
  %%their corresponding function values (from greatest/worst to smallest/
  %%best)
    Order = fx;

end
function orderx = OrderX(fnew,ln)
%%The purpose of this function is to separate the function values and their
%%corresponding points in the ordered matrix fnew

%%lc denotes the number of columns in fnew which allows us to do our
%%separation since we know that the last column contains the function
%%values of each vertex
lc = length(fnew(1,:));
    for i = 1:ln
        for j = 1:lc-1
            xn(i,j)=fnew(i,j);
        
        end
    end
%%Now we may output the matrix containing the vertices in decending order
%%of function value
orderx=xn;
end
function ref = Reflection(xnew,ln)
%%I generate an empty row with n-1 columns
    for j=1:ln-1
       sumx(1,j)=0; 
    end
%%Now i use my empty row to generate the sum of all my points, excluding of
%%course my worst vertex
      
    for i = 2:ln
       sumx= sumx + xnew(i,:) ;
    end
    %%Now i define the centroid G, of all vertices except the worst vertex
    G= sumx/(ln-1);
    
    %%Now we are able to return the reflection of worst vertex
    
    ref = 2*G-xnew(1,:);


end
function exp = Expand(xbar,xnew,a,ln)
%%Note: this function is used for both expansion and contraction and allows
%%us to switch between the two through its input variable a (i.e. for
%%expansion, a = 2 and contraction, a = 0.5
    
    %%The below for loop generates a zero vector with the same number of
    %%colums as the coordinates of our vertices. This is done so that in
    %%the next for loop, i can successfully do the sigma (summation) of all
    %%the vertices which is needed to calculate the centroid G
    for j=1:ln-1
       sumx(1,j)=0; 
    end
    for i = 2:ln
       sumx= sumx + xnew(i,:) ;
    end
    %%G denotes the centroid 
     G= sumx/(ln-1);
    %%Finally we are able to output our expansion/contraction factor
     exp=a*xbar +(1-a)*G;
       
end
function var= StopCriteria(xnew,ln)
%%Since the stopping criteria provided was a function that resembled the
%%variance function, I treated it as such and hence the use of variable
%%names such as var (variance) and mean 

sum=0;
    for i=1:ln
      sum= sum+RunF(xnew(i,:));  
    end
 mean= sum/ln;
 x=0;
 for j=1:ln
    x=x+(RunF(xnew(j,:))-mean)^2;
 end
 var=(x/(ln+1))^(0.5);
 
end