function Lu = Laplace2d(u,nx,ny,Dx,Dy)

% set the shape of un
Lu=zeros(size(u));

for j = 1:ny
    for i = 1:nx
        if(i>2 && i<nx-1 && j>2 && j<ny-1)
            Lu(i,j) = ...
                Dx/12*(-u(i+2,j)+16*u(i+1,j)-30*u(i,j)+16*u(i-1,j)-u(i-2,j)) + ...
                Dy/12*(-u(i,j+2)+16*u(i,j+1)-30*u(i,j)+16*u(i,j-1)-u(i,j-2));
        else
            Lu(i,j) = 0.0;
        end
    end
end
