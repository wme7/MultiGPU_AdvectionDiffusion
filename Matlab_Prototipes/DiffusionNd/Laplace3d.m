function Lu = Laplace3d(u,nx,ny,nz,Dx,Dy,Dz)

% set the shape of un
Lu=zeros(size(u)); 

strategy = 'OMP';
switch strategy
    case 'CPU'
        for k = 1:nz
            for j = 1:ny
                for i = 1:nx
                    
                    %o = i + nx*j + xy*k; % node( j,i,k )      n  b
                    %n = o + nx;          % node(j+1,i,k)      | /
                    %s = o - nx;          % node(j-1,i,k)      |/
                    %e = o + 1;           % node(j,i+1,k)  w---o---e
                    %w = o - 1;           % node(j,i-1,k)     /|
                    %t = o + xy;          % node(j,i,k+1)    / |
                    %b = o - xy;          % node(j,i,k-1)   t  s
                    
                    if(i>2 && i<nx-1 && j>2 && j<ny-1 && k>2 && k<nz-1)
                        Lu(i,j,k) = ...
                            Dx/12*(-u(i+2,j,k)+16*u(i+1,j,k)-30*u(i,j,k)+16*u(i-1,j,k)-u(i-2,j,k)) + ...
                            Dy/12*(-u(i,j+2,k)+16*u(i,j+1,k)-30*u(i,j,k)+16*u(i,j-1,k)-u(i,j-2,k)) + ...
                            Dz/12*(-u(i,j,k+2)+16*u(i,j,k+1)-30*u(i,j,k)+16*u(i,j,k-1)-u(i,j,k-2));
                    else
                        Lu(i,j,k) = 0.0;
                    end
                end
            end
        end
    case 'OMP'
        xy = nx*ny; nx2 = 2*nx; xy2 = 2*xy;
        
        for j = 1:ny
            for i = 1:nx
                % Single index
                o = i+nx*(j-1)+xy*2;
                
                if (i>2 && i<nx-1 && j>2 && j<ny-1)
                    below2=u(o-xy2); below=u(o-xy); center=u(o); above=u(o+xy); above2=u(o+xy2);
                    
                    Lu(o) =  ...
                        Dx/12*(-  u(o-2)+16*u(o-1) -30*center+16*u(o+1) -u(o+2)  )+ ...
                        Dy/12*(-u(o-nx2)+16*u(o-nx)-30*center+16*u(o+nx)-u(o+nx2))+ ...
                        Dz/12*(- below2 + 16*below -30*center+16* above - above2 );
                    
                    for k = 3:nz-3
                        o=o+xy; below2=u(o-xy2); below=u(o-xy); center=u(o); above=u(o+xy); above2=u(o+xy2);
                        
                        Lu(o) = ...
                            Dx/12*(-  u(o-2)+16*u(o-1) -30*center+16*u(o+1) -u(o+2)  )+ ...
                            Dy/12*(-u(o-nx2)+16*u(o-nx)-30*center+16*u(o+nx)-u(o+nx2))+ ...
                            Dz/12*(- below2 + 16*below -30*center+16* above - above2 );
                    end
                else
                    Lu(o) = 0.0;
                end
            end
        end
end