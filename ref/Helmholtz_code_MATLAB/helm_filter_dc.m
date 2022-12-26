function [c, dctil, xtil] = helm_filter_dc(node, elem, numnode, numelem, x, U, k, penal, rmin, s, t)
% DECLARE LOCAL VARIABLE MATRIX
localnode = zeros(4,2); localelem = zeros(1,4);
% DECLARE HELMHOLTZ VARIABLE VECTOR
xtil = zeros(numelem,1); A = zeros(numnode,numnode); b = zeros(numnode,1);
% DECLARE SENSITIVITY VARIABLE VECTOR
c = 0; dctil = zeros(numelem,1); Ue = zeros(8,1);
Asen = zeros(4,4); bsen = zeros(4,1); dce = zeros(1,4);
% RESCALING VARIABLE VECTOR UP TO EACH NODE
xups = repmat(x/4,1,4);
% BVP HELMHOLTZ FILTER USING MATRIX NOTATION
for i = 1:numelem % FOR EACH LOCAL ELEMENT
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
% SOLVING SENSITIVITY W.R.T FILTERING VARIABLE
for i = 1:numelem % FOR EACH LOCAL ELEMENT
    for ii = 1:4
        localnode(ii,1) = node(elem(i,ii+1),2); % x NODE POSITION
        localnode(ii,2) = node(elem(i,ii+1),3); % y NODE POSITION
        localelem(ii) = elem(i,ii+1); % NODE NUMBERING INDEX
        Ue(2*ii-1,1) = U(2*elem(i,ii+1)-1);
        Ue(2*ii,1) = U(2*elem(i,ii+1));
    end
    % DETERMINE THE VALUE OF C, DC TILDA FOR EACH NODE
    c = c+(xtil(i)^penal)*Ue'*k(:,:,i)*Ue;
    for ii = 1:4
        [Jsen] = jacobmat(localnode, s(ii), t(ii)); detJsen = Jsen(1,1)*Jsen(2,2)-Jsen(1,2)*Jsen(2,1);
        % RECALL SHAPE FUNCTION
        N1sen = (1-s(ii))*(1-t(ii))/4;   N2sen = (1+s(ii))*(1-t(ii))/4;
        N3sen = (1+s(ii))*(1+t(ii))/4;   N4sen = (1-s(ii))*(1+t(ii))/4;
        Nsen = [N1sen,N2sen,N3sen,N4sen]; trNsen = [N1sen;N2sen;N3sen;N4sen];
        gradNsen = 0.25*[-(1-t(ii)),(1-t(ii)),(1+t(ii)),-(1+t(ii));...
                         -(1-s(ii)),-(1+s(ii)),(1+s(ii)),(1-s(ii))];
        trgradNsen = 0.25*[-(1-t(ii)),-(1-s(ii));(1-t(ii)),-(1+s(ii));...
                            (1+t(ii)),(1+s(ii));-(1+t(ii)),(1-s(ii))];
        % SOLVING SENSITIVITY INTEGRAL TERM
        Asen(:,:) = Asen(:,:)+(rmin^2)*trgradNsen*gradNsen+trNsen*Nsen*detJsen;
        bsen(:,1) = bsen(:,1)+trNsen;
        dce(1,:) = dce(1,:)+-penal*nodextil(localelem(ii))^(penal-1)*Ue'*k(:,:,i)*Ue*Nsen;
    end
    dctil(i) = dce/Asen*bsen;
end