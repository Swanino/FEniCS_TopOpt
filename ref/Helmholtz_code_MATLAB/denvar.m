function cgx = denvar(elmx, elmy, volfrac, rmin)
ix = ones(elmy*elmx*(2*(ceil(rmin)-1)+1)^2,1);
jx = ones(size(ix));
sx = zeros(size(ix));
m = 0;
for i1 = 1:elmx
    for j1 = 1:elmy
        e1 = (i1-1)*elmy+j1;
        for i2 = 1:2
            for j2 = 1:2
                e2 = (i2-1)*2+j2;
                m = m+1;
                ix(m) = e1;
                jx(m) = e2;
                sx(m) = volfrac/4;
            end
        end
    end
end
cgx = sparse(ix,jx,sx);