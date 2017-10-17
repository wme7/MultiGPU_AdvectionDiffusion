% Produce Run

% =========================== HeatEq3D (13-pt) =======================
% Optimization                                 :  none
% Kernel time ex. data transfers               :  5.810000 seconds
% ===================================================================
% Total effective GFLOPs                       :  6.664205
% ===================================================================
% 3D Grid Size                                 :  201 x 199 x 200 
% Iterations                                   :  605
% Final Time                                   :  0.329157
% ===================================================================

% Set run parameters
C = 1.0;
L = 10;
W = 10;
H = 10;
nx = 201;
ny = 199;
nz = 200;
iter = 605;
block_X = 32;
block_Y = 4;
block_Z = 4;

% Write sh.run
fID = fopen('run.sh','wt');
fprintf(fID,'make\n');
args = sprintf('%1.1f %1.2f %1.2f %1.2f %d %d %d %d %d %d %d',C,L,W,H,nx,ny,nz,iter,block_X,block_Y,block_Z);
fprintf(fID,'./diffusion3d.run %s\n',args);
fclose(fID);

% Execute sh.run
! sh run.sh

% Load data
fID = fopen('result.bin');
output = fread(fID,[1,nx*ny*nz],'float')';

% Plot data
% myplot(output,nx,ny,nz,L,W,H);
p = reshape(output,nx,ny,nz);

[x,y,z]=meshgrid(linspace(-W/2,W/2,ny),linspace(-L/2,L/2,nx),linspace(-H/2,H/2,nz));

figure(1); h1=slice(x,y,z,p,0,0,0); 
cm=getColorMap('kwave'); colormap(cm); axis tight; colorbar; caxis([-1,1]);
h1(1).EdgeColor = 'none'; xlabel('y'); 
h1(2).EdgeColor = 'none'; ylabel('x'); 
h1(3).EdgeColor = 'none'; zlabel('z'); 
title('Diffusion-3D GPU: Temperature field.');
print('diffusion3d.png','-dpng')

% Clean up
! rm -rf *.bin
 