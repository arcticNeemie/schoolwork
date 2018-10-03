function ans = typeChecker(x)
r = size(x,1);
c = size(x,2);
if r==0 && c==0
    ans =-1;
elseif r>=1 || c>=1
    ans = 1;
elseif r==1 && c==1
    ans=0;
else
   ans=2;
end
    
if ischar(x) || istable(x)
   ans = 2; 
end

end