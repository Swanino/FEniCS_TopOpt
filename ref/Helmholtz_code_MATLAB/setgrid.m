function [node, elem] = setgrid(nelx, nely, elmx, elmy)
% ----------------- NODE & ELEMENT INFO -----------------
%  (elmy+1)*(i-1)+j+1 ----------------- (elemy+1)*i+j+1
%          |          (elmx*elmy)*i+j)         |
%  (elmy+1)*(i-1)+j ------------------- (elemy+1)*i+j
numnode = (elmx+1)*(elmy+1); numelm = elmx*elmy;
xposvec = linspace(0,nelx,elmx+1); yposvec = linspace(0,nely,elmy+1);
[x,y] = meshgrid(xposvec,yposvec);
node = zeros(numnode,3); node(:,1) = 1:numnode;
node(:,2) = x(:); node(:,3) = y(:);
elem = zeros(numelm,5); elem(:,1) = 1:numelm;
for i = 1:elmx
    for j = 1:elmy
        flag = elmy*(i-1)+j;
        elem(flag,1) = flag;
        elem(flag,2) = (elmy+1)*(i-1)+j;
        elem(flag,3) = (elmy+1)*i+j;
        elem(flag,4) = (elmy+1)*i+j+1;
        elem(flag,5) = (elmy+1)*(i-1)+j+1;
    end
end