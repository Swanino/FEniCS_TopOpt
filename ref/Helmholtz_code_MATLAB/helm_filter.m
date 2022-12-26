function xtil = helm_filter(node, elem, ndof, x, rmin, s, t)
% DECLARE HELMHOLTZ VARIABLE VECTOR
localnode = zeros(4,2); localelem = zeros(1,4);
A = zeros(ndof,ndof); b = zeros(ndof,1); xtil = zeros(length(x),1);
% DECLARE SENSITIVITY VARIABLE VECTOR
% RESCALING VARIABLE VECTOR UP TO EACH NODE
%xups = repmat(x/4,1,4);

ckdup = elem(:,2:5); dup = zeros(length(ckdup),4);
for i = 1:ndof
    [m,n] = find(ckdup==i);
    for ii = 1:length(m)
        dup(m(ii),n(ii))=2*length(m);
    end
end
xups = zeros(length(x),4);
for i = 1:length(dup)
    for ii = 1:4
        xups(i,ii) = x(i)/dup(i,ii);
    end
end

% elmx=60; elmy=20; rdash=1.5;
% iH = ones(elmy*elmx*(2*(ceil(rdash)-1)+1)^2,1); jH = ones(size(iH)); sH = zeros(size(iH));
% k = 0;
% for i1 = 1:elmx
%     for j1 = 1:elmy
%         e1 = (i1-1)*elmy+j1;
%         for i2 = max(i1-(ceil(rdash)-1),1):min(i1+(ceil(rdash)-1),elmx)
%             for j2 = max(j1-(ceil(rdash)-1),1):min(j1+(ceil(rdash)-1),elmy)
%                 e2 = (i2-1)*elmy+j2;
%                 k = k+1;
%                 iH(k) = e1;
%                 jH(k) = e2;
%                 sH(k) = max(1,2*((i1-i2)^2+(j1-j2)^2));
%             end
%         end
%     end
% end
% H = sparse(iH,jH,sH); Hfull = full(H);
% xups = mean(x,'all')./Hfull;

% BVP HELMHOLTZ FILTER USING MATRIX NOTATION
for i = 1:length(x) % FOR EACH LOCAL ELEMENT
    for ii = 1:4
        localnode(ii,1) = node(elem(i,ii+1),2); % x NODE POSITION
        localnode(ii,2) = node(elem(i,ii+1),3); % y NODE POSITION
        localelem(ii) = elem(i,ii+1); % NODE NUMBERING INDEX
    end
    for ii = 1:4
        [J] = jacobmat(localnode, s(ii), t(ii)); detJ = J(1,1)*J(2,2)-J(1,2)*J(2,1);
        % DETERMINE SHAPE FUNCTION
        N1 = (1-s(ii))*(1-t(ii))/4;   N2 = (1+s(ii))*(1-t(ii))/4;
        N3 = (1+s(ii))*(1+t(ii))/4;   N4 = (1-s(ii))*(1+t(ii))/4;
        N = [N1,N2,N3,N4]; trN = [N1;N2;N3;N4];
        gradN = 0.25*[-(1-t(ii)),(1-t(ii)),(1+t(ii)),-(1+t(ii));...
                      -(1-s(ii)),-(1+s(ii)),(1+s(ii)),(1-s(ii))];
        trgradN = 0.25*[-(1-t(ii)),-(1-s(ii));(1-t(ii)),-(1+s(ii));...
                         (1+t(ii)),(1+s(ii));-(1+t(ii)),(1-s(ii))];
        % SOLVING HELMHOLTZ FILTER INTEGRAL TERM
        A(localelem,localelem) = A(localelem,localelem)+((rmin^2)*trgradN*gradN+trN*N*detJ);
        b(localelem,1) = b(localelem,1)+xups(i,ii)*trN;
    end
end
% DETERMINE THE VALUE OF X TILDA FOR EACH NODE
%c = [0.25;0.125;0.25;0.125;0.0625;0.125;0.25;0.125;0.25];
nodextil = A\b;
% RESCALING VARIABLE VECTOR DOWN TO EACH ELEMENT
for i = 1:length(x)
    for ii = 1:4
        localelem(ii) = elem(i,ii+1); % NODE NUMBERING INDEX
    end
    for ii = 1:4
        xtil(i) = xtil(i)+nodextil(localelem(ii));
    end
end