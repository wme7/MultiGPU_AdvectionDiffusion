% Produce Run
close all;

% ===========================Burgers-2D MPI-GPU-WENO5=======================
% Optimization                                 :  Pitched Memory
% Kernel time ex. data transfers               :  6.291269 seconds
% Data transfer(s) HtD                         :  0.000475 seconds
% Data transfer(s) DtH                         :  0.000484 seconds
% ===================================================================
% Total effective GFLOPs                       :  0.123905
% ===================================================================
% 3D Grid Size                                 :  400 x 406
% Iterations                                   :  200 x 3 RK steps
% ===================================================================

% Set run parameters
tEnd=0.40;
CFL= 0.40;
L = 2.0;
W = 2.0;
nx = 400;
ny = 400;
block_X = 32;
block_Y = 32;
np = 2;
RADIUS=3;
HALO=2*RADIUS;

% Write sh.run
fID = fopen('run.sh','wt');
fprintf(fID,'make\n');
args = sprintf('%1.2f %1.2f %1.2f %1.2f %d %d %d %d',tEnd,CFL,L,W,nx,ny,block_X,block_Y);
fprintf(fID,'mpirun -np %d Burgers2d.run %s\n',np,args); 
%profile = 'nvprof -f -o Burgers3d.%q{OMPI_COMM_WORLD_RANK}.nvprof';
%fprintf(fID,'mpirun -np %d %s ./Burgers3d.run %s\n',np,profile,args);
fclose(fID);

% Execute sh.run
! sh run.sh

% Build discrete domain
dx=L/(nx-1); xc=-L/2:dx:L/2;
dy=W/(ny-1); ny=ny+HALO; yc=-W/2-RADIUS*dy:dy:W/2+RADIUS*dy;
[x,y]=meshgrid(yc,xc);

% Set plot region
region = [-L/2,L/2,-W/2,W/2,0,1]; 

%% Load and plot data
fID = fopen('result.bin');
output = fread(fID,[1,nx*ny],'float')';
u = reshape(output,nx,ny);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(1)
surf(x,y,u,'Edgecolor','none'); axis(region); colormap(getColorMap('kwave')); view(90,90);
title('Inviscid Burgers, MultiGPU-WENO5-RK3','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
print('InviscidBurgers2d','-dpng');

%% Load and plot Initial Condition
fID = fopen('initial.bin');
output = fread(fID,[1,nx*ny],'float')';
u0 = reshape(output,nx,ny);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(2)
surf(x,y,u0,'Edgecolor','none'); axis(region); colormap(getColorMap('kwave')); view(90,90);
title('Initial Condition','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
print('InitialCondition2d','-dpng');

% Clean up
! rm -rf *.bin 
