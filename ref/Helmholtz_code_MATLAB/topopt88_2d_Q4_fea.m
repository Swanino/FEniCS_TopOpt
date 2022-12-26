%%%% AN 88 LINE TOPOLOGY OPTIMIZATION CODE %%%%
function topopt88_2d_Q4(nelx, nely, elmx, elmy, volfrac, penal, rmin, ft)
%% PREPARE FINITE ELEMENT ANALYSIS
nelx=60;nely=20;elmx=60;elmy=20;volfrac=0.75;penal=3;rmin=1.5;ft=1;
% GIVE NODE & ELEMENT XY POSITION EACH NODE POINT
[node, elem] = setgrid(nelx, nely, elmx, elmy);
numnode = size(node,1); numelem = size(elem,1);
% BOUNDARY CONDITIONS
xlowidx = abs(node(:,2)) < 1e-9; xhighidx = abs(node(:,2)-max(node(:,2))) < 1e-9;
ylowidx = abs(node(:,3)) < 1e-9; yhighidx = abs(node(:,3)-max(node(:,3))) < 1e-9;
% DECLARE BOUNDARY POSITION -> X DIR. LEFT SIDES & RIGHT BOTTOM EDGE POINT
fixeddofs = zeros(numnode,2); fixeddofs(xlowidx,1) = 1; fixeddofs(xhighidx&ylowidx,2) = 1;
freedofs = find(reshape(~fixeddofs',[],1));
% DEFINE LOADS AND SUPPORTS (HALF MDD-BEAM, DISPLACEMENT -> 2n-1: u VECTOR / 2n: v VECTOR)
loadpos = zeros(numnode,2); loadpos(xlowidx&yhighidx,2) = 1;
loadposidx = find(reshape(loadpos',[],1));
F = zeros(2*(nely+1)*(nelx+1),1);
for i = 1:size(loadposidx,1)
    F(loadposidx(i)) = -1e2;
end
U = zeros(2*(nelx+1)*(nely+1),1);
%% PREPARE FILTER
iH = ones(elmy*elmx*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
m = 0;
for i1 = 1:elmx
    for j1 = 1:elmy
        e1 = (i1-1)*elmy+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),elmx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),elmy)
                e2 = (i2-1)*elmy+j2;
                m = m+1;
                iH(m) = e1;
                jH(m) = e2;
                sH(m) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);
%% SET STIFFNESS MATRIX
% MATERIAL PROPERTIES
E0 = 200e3; nu = 0.25; h = 1;
% DEFINE STRESS-STRAIN MATRIX
D = (E0/(1-nu^2)) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
% S-T AXIS
s = 1/sqrt(3) * [-1,1,1,-1]; t = 1/sqrt(3) * [-1,-1,1,1];
% VARIABLE OF VOLUME FRACTION
x = repmat(volfrac,elmy,elmx);
%% INITIALIZE & START ITERATION
xPhys = x; Ue = zeros(8,1); dc = zeros(elmy,elmx); loop = 0; change = 1;
while change > 0.01
    loop = loop + 1;
    %% FE-ANALYSIS
    % DETERMINE STIFFNESS MATRIX
    [K,k] = stiffmat(node, elem, numnode, numelem, h, D, s, t, x, penal);
    K = sparse(K);
    % SOLVING DISPLACEMENT VECTOR
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    c = 0;
    for i = 1:numelem
        for j = 1:4
            Ue(2*j-1,1) = U(2*elem(i,j+1)-1);
            Ue(2*j,1) = U(2*elem(i,j+1));
        end
        c = c+(x(i)^penal)*Ue'*k(:,:,i)*Ue;
        dc(i) = -penal*x(i)^(penal-1)*Ue'*k(:,:,i)*Ue;
    end
    dv = ones(elmy,elmx);
    %% FILTERING/MODIFICATION OF SENSITIVITIES
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    end
    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    l1 = 0; l2 = 1e5; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-6
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid))))); xnew(xnew <= 1e-5) = 1e-5;
        if ft == 1
            xPhys = xnew;
        elseif ft == 2
            xPhys(:) = (H*xnew(:))./Hs;
        end
        if sum(xPhys(:)) > volfrac*elmy*elmx, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;
    %% PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c,mean(xPhys(:)),change);
    %% PLOT DENSITIES
    colormap(gray); imagesc(flipud(1-xPhys)); caxis([0 1]); axis equal; axis off; drawnow;
end