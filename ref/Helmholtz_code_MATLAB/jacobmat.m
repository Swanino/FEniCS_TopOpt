function [J] = jacobmat(node, s, t)
% DETERMINE JACOBIAN ELEMENT
j11 = 0.25*(-(1-t)*node(1,1)+(1-t)*node(2,1)+(1+t)*node(3,1)-(1+t)*node(4,1));
j12 = 0.25*(-(1-t)*node(1,2)+(1-t)*node(2,2)+(1+t)*node(3,2)-(1+t)*node(4,2));
j21 = 0.25*(-(1-s)*node(1,1)-(1+s)*node(2,1)+(1+s)*node(3,1)+(1-s)*node(4,1));
j22 = 0.25*(-(1-s)*node(1,2)-(1+s)*node(2,2)+(1+s)*node(3,2)+(1-s)*node(4,2));
% DETERMINE JACOBIAN MATRIX
J = [j11,j12; j21,j22];