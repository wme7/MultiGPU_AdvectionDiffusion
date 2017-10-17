function myplot(result,nx,ny,L,W)

[x,y] = meshgrid(linspace(-L/2,L/2,nx),linspace(-W/2,W/2,ny));
S = reshape(result(:,1),nx,ny);

h = surf(x,y,S,'EdgeColor','none'); %colormap hot;
%axis equal;
axis tight;
view(-45,45)

print('diffusion2d','-dpng')