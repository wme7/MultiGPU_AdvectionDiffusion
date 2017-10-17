function myplot(result,nx,ny,nz,L,W,H)

[x,y,z] = meshgrid(linspace(-L/2,L/2,nx),linspace(-W/2,W/2,ny),linspace(-H/2,H/2,nz));
V = reshape(result(:,1),nx,ny,nz);

h = slice(x,y,z,V,0,0,0);
h(1).EdgeColor = 'none';
h(2).EdgeColor = 'none';
h(3).EdgeColor = 'none';
%colormap jet;
axis equal;
axis tight;
colorbar;

print('diffusion3d','-dpng')