function Lu = Laplace2d_axisymmetric(u,nr,ny,Dr,Dy,dr,inv_r)

% set the shape of un
Lu=zeros(size(u));

for j = 1:ny
    for i = 1:nr
        if(i>2 && i<nr-1 && j>2 && j<ny-1)
            Lu(j,i) = ...
                Dr*(-u(j,i+2)+16*u(j,i+1)-30*u(j,i)+16*u(j,i-1)-u(j,i-2)) + ...
                Dy*(-u(j+2,i)+16*u(j+1,i)-30*u(j,i)+16*u(j-1,i)-u(j-2,i)) + ...
                Dr*dr*inv_r(i)*(u(j,i-2)-8*u(j,i-1)+8*u(j,i+1)-u(j,i+2));
        else
            Lu(j,i) = 0.0;
        end
    end
end
