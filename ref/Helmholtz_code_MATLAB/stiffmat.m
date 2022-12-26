function [K,k] = stiffmat(node, elem, numnode, numelem, h, D, s, t, x, penal)
% DECLARE LOCALNODE VECTOR & NUMBER OF DOF PER EACH ELEMENT
localelem = zeros(1,4); dofpe = 2;
K = zeros(numnode*dofpe,numnode*dofpe);
k = zeros(8,8,numelem);
for i = 1:numelem
    knpe = zeros(8,8);
    for ii = 1:4
        localelem(ii) = elem(i,ii+1);
    end
    for ii = 1:4
        [B,detJ] = strdisjacob(node, elem(i,:), s(ii), t(ii));
        knpe(:,:) = knpe(:,:)+B'*D*B*detJ*h;
    end
    k(:,:,i) = knpe(:,:);
    m = 0;
    for ii = 1:4
        start = (localelem(ii)-1)*dofpe;
        for jj = 1:dofpe
            m = m+1;
            elemidx(m) = start+jj;
        end
    end
    % ASSEMBLE EACH ELEMENT STIFFNESS MATRIXES
    for ii = 1:length(elemidx)
        m = elemidx(ii);
        for jj = 1:length(elemidx)
            n = elemidx(jj);
            K(m,n) = K(m,n)+(x(i)^penal)*(knpe(ii,jj));
        end
    end
end