% Produce Run
close all;

% ===========================Diffusion-2D MPI-GPU-FD4=======================
% Optimization                                 :  Pitched Memory
% Kernel time ex. data transfers               :  0.636297 seconds
% Data transfer(s) HtD                         :  0.000653 seconds
% Data transfer(s) DtH                         :  0.000485 seconds
% ===================================================================
% Total effective GFLOPs                       :  6.131567
% ===================================================================
% 3D Grid Size                                 :  400 x 406
% Iterations                                   :  1001 x 3 RK steps
% ===================================================================

% Set run parameters
K = 1.0;
L = 2.0;
W = 2.0;
nx = 400;
ny = 400;
iter = 1000;
block_X = 32;
block_Y = 32;
np = 2;
RADIUS = 3;

% Write sh.run
fID = fopen('run.sh','wt');
fprintf(fID,'make\n');
args = sprintf('%1.2f %1.2f %1.2f %d %d %d %d %d',K,L,W,nx,ny,iter,block_X,block_Y);
fprintf(fID,'mpirun -np %d Diffusion2d.run %s\n',np,args); 
%profile = 'nvprof -f -o Diffusion3d.%q{OMPI_COMM_WORLD_RANK}.nvprof';
%fprintf(fID,'mpirun -np %d %s ./Diffusion3d.run %s\n',np,profile,args);
fclose(fID);

% Execute sh.run
! sh run.sh

% Build discrete domain
dx=L/(nx-1); xc=-L/2:dx:L/2;
dy=W/(ny-1); ny=ny+2*RADIUS; yc=-W/2-RADIUS*dy:dy:W/2+RADIUS*dy;
[x,y]=meshgrid(yc,xc);

% Set plot region
region = [-L/2,L/2,-W/2,W/2,0,1]; 

%% Load and plot Initial Condition
fID = fopen('initial.bin');
output = fread(fID,[1,nx*ny],'float')';
u0 = reshape(output,nx,ny);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(1)
surf(x,y,u0,'Edgecolor','none'); axis(region);
title('Initial Condition','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
print('InitialCondition2d','-dpng');


%% Load and plot data
fID = fopen('result.bin');
output = fread(fID,[1,nx*ny],'float')';
u = reshape(output,nx,ny);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(2)
surf(x,y,u,'Edgecolor','none'); axis(region);
title('Heat Equation, MultiGPU-FDM4-RK3','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
print('InviscidBurgers2d','-dpng');

% Clean up
! rm -rf *.bin 
