function [B,detJ] = strdisjacob(node, elem, s, t)
% PLEASE ENTER LOCAL ELEMENT VECTOR (EX. elem(NUMBER,:)...)
% ALSO, ENTER S, T COORDINATES PARAMETERS AS MATRIX ELEMENT SCALAR VALUE. NOT MATRIX!!
localnode = zeros(4,2); B = zeros(3,8);
% INDEXING LOCAL NODE VALUE IN EACH DOMAIN
for i = 1:4
    localnode(i,1) = node(elem(:,i+1),2);
    localnode(i,2) = node(elem(:,i+1),3);
end
% DETERMINE DETERMINANT JACOBIAN
[J] = jacobmat(localnode, s, t);
detJ = det(J);
% DETERMINE SHAPE FUNCTION
N(1,1) = (t-1)/4;  N(1,2) = (s-1)/4;
N(2,1) = (1-t)/4;  N(2,2) = -(1+s)/4;
N(3,1) = (t+1)/4;  N(3,2) = (s+1)/4;
N(4,1) = -(t+1)/4; N(4,2) = (1-s)/4;
% DETERMINE DUMMY VARIABLE
a = 0.25*(localnode(1,2)*(s-1)+localnode(2,2)*(-1-s)+localnode(3,2)*(1+s)+localnode(4,2)*(1-s));
b = 0.25*(localnode(1,2)*(t-1)+localnode(2,2)*(1-t)+localnode(3,2)*(1+t)+localnode(4,2)*(-1-t));
c = 0.25*(localnode(1,1)*(t-1)+localnode(2,1)*(1-t)+localnode(3,1)*(1+t)+localnode(4,1)*(-1-t));
d = 0.25*(localnode(1,1)*(s-1)+localnode(2,1)*(-1-s)+localnode(3,1)*(1+s)+localnode(4,1)*(1-s));
% DETERMINE STRAIN/DISPLACEMENT MATRIX
for i = 1:4
    B(1,2*i-1) = a*N(i,1)-b*N(i,2);
    B(2,2*i) = c*N(i,2)-d*N(i,1);
    B(3,2*i-1) = B(2,2*i);
    B(3,2*i) = B(1,2*i-1);
end
B = B/detJ;