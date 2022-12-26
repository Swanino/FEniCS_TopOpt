%%%% AN 88 LINE TOPOLOGY OPTIMIZATION CODE %%%%
function topopt88_2d_Q4(nelx, nely, elmx, elmy, volfrac, penal, rmin)
%% PREPARE FINITE ELEMENT ANALYSIS
nelx=60;nely=20;elmx=60;elmy=20;volfrac=0.5;penal=3;rmin=3/2/sqrt(3);
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
    F(loadposidx(i)) = -1;
end
U = zeros(2*(nelx+1)*(nely+1),1);
%% PREPARE VARIABLE X FOR HF FILTER (SET DENSITY DIMENSIONS CG: Lagrange / DG: Discrete Galerkin)
cgx = denvar(elmx, elmy, volfrac, rmin); dgx = sum(cgx,2); dgxidx = full(flipud(reshape(dgx,[elmy,elmx])));
%% SET STIFFNESS MATRIX
% MATERIAL PROPERTIES
E0 = 1; nu = 0.3; h = 1;
% DEFINE STRESS-STRAIN MATRIX
D = (E0/(1-nu^2)) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
% S-T AXIS
s = 1/sqrt(3) * [-1,1,1,-1]; t = 1/sqrt(3) * [-1,-1,1,1];
%% INITIALIZE & START ITERATION
xPhys = dgxidx; dc = zeros(elmy,elmx); Ue = zeros(8,1); loop = 0; change = 1; pltlpc = zeros(1,50);
% PLOT RESULT
f1 = figure('Name','Data plot chart'); f2 = figure('Name', 'Density simulation results');
while change > 0.01
%while change > 0.01
    loop = loop + 1;
    %% FE-ANALYSIS
    xPhystp = reshape(flipud(xPhys),[elmy*elmx,1]); % MAKE COLUMN VECTOR SHAPE MATRIX(TO CALCULATE)
    % DETERMINE STIFFNESS MATRIX
    [K,k] = stiffmat(node, elem, numnode, numelem, h, D, s, t, xPhystp, penal); K = sparse(K);
    % SOLVING DISPLACEMENT VECTOR
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    c = 0;
    for i = 1:numelem
        for j = 1:4
            Ue(2*j-1,1) = U(2*elem(i,j+1)-1);
            Ue(2*j,1) = U(2*elem(i,j+1));
        end
        c = c+(xPhystp(i)^penal)*Ue'*k(:,:,i)*Ue;
        dc(i) = -penal*xPhystp(i)^(penal-1)*Ue'*k(:,:,i)*Ue;
    end
    dc = flipud(dc); dv = ones(elmy,elmx);
    %% FILTERING/MODIFICATION OF SENSITIVITIES
    dgxtil = helm_filter(node, elem, numnode, xPhystp, rmin, s, t);
    % RESHAPE MATRIX TO ELEMENT SHAPE & FILTERING
    dgxtil = reshape(dgxtil,[elmy,elmx]); dgxtil = flipud(dgxtil);
    dc(:) = (dgxtil(:).*dc(:).*xPhys(:))./max(1e-3,xPhys(:));
    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(dgxidx-move,min(1,min(dgxidx+move,dgxidx.*sqrt(-dc./dv/lmid))))); xnew(xnew <= 1e-3) = 1e-3;
        xPhys = xnew;
        if sum(xPhys(:)) > volfrac*elmy*elmx, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-dgxidx(:))); dgxidx = xnew;
    %% PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c,full(mean(xPhys(:))),change);
    pltlpc(loop) = c;
    figure(f1); plot(1:50,pltlpc,'or'); xlabel('iteration'); ylabel('compliance'); drawnow;
    %% PLOT DENSITIES
    figure(f2); colormap(gray); imagesc(1-xPhys); caxis([0 1]); axis equal; axis off; drawnow;
end