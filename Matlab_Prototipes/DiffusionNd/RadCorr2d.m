function Lu_new = RadCorr2d(u,nr,ny,Dr,dr,inv_r)
% Radial Correction factor: 1/r*(du/dr)

% set the shape of un
Lu_new=zeros(size(u));

for j = 1:ny
    for i = 1:nr
        if(i>2 && i<nr-1 && j>2 && j<ny-1)
            Lu_new(i,j) = ...
                Dr*dr*inv_r(i)/12*(u(i+2,j)-8*u(i+1,j)+8*u(i-1,j)-u(i-2,j));
        else
            Lu_new(i,j) = 0.0;
        end
    end
end
