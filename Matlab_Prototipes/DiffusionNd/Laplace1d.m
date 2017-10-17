function d2udx2 = Laplace1d(u,nx,Dx)

% set the shape of un
d2udx2=zeros(size(u));

for i = 1:nx
    if(i>2 && i<nx-1)
        d2udx2(i)= Dx/12*(-u(i+2)+16*u(i+1)-30*u(i)+16*u(i-1)-u(i-2));
    else
        d2udx2(i)= 0;
    end
end
