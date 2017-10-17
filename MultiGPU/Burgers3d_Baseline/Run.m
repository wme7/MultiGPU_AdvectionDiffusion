% Produce Run
close all;

% =======================Burgers-3D MPI-GPU-WENO5=====================
% Optimization                                 :  Pitched Memory
% Kernel time ex. data transfers               :  1373.254345 seconds
% Data transfer(s) HtD                         :  0.172002 seconds
% Data transfer(s) DtH                         :  0.158721 seconds
% ===================================================================
% Total effective GFLOPs                       :  0.303122
% ===================================================================
% 3D Grid Size                                 :  400 x 400 x 406
% Iterations                                   :  267 x 3 RK steps
% ===================================================================

% Set run parameters
tEnd=0.40;
CFL= 0.30;
L = 2.0;
W = 2.0;
H = 4.0;
nx = 200;
ny = 200;
nz = 200;
block_X = 8;
block_Y = 8;
block_Z = 8;
np = 2;
RADIUS=3;
HALO=2*RADIUS;

% Write sh.run
fID = fopen('run.sh','wt');
fprintf(fID,'make\n');
args = sprintf('%1.2f %1.2f %1.2f %1.2f %1.2f %d %d %d %d %d %d',tEnd,CFL,L,W,H,nx,ny,nz,block_X,block_Y,block_Z);
fprintf(fID,'mpirun -np %d Burgers3d.run %s\n',np,args); 
%profile = 'nvprof -f -o Burgers3d.%q{OMPI_COMM_WORLD_RANK}.nvprof';
%fprintf(fID,'mpirun -np %d %s ./Burgers3d.run %s\n',np,profile,args);
fclose(fID);

% Execute sh.run
! sh run.sh

% Build discrete domain
dx=L/(nx-1); xc=-L/2:dx:L/2;
dy=W/(ny-1); yc=-W/2:dy:W/2;
dz=H/(nz-1); nz=nz+HALO; zc=-H/2-RADIUS*dz:dz:H/2+RADIUS*dz;
[x,y,z]=meshgrid(yc,xc,zc);

% Set plot region
region = [-L/2,L/2,-W/2,W/2,-H/2,H/2]; 

% Load and plot data
fID = fopen('result.bin');
output = fread(fID,[1,nx*ny*nz],'float')';
u = reshape(output,nx,ny,nz);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(1)
q=slice(x,y,z,u,1/3,1/3,1/3); axis(region); colormap(getColorMap('kwave')); view(90,90);
title('Inviscid Burgers, MultiGPU-WENO5-RK3','interpreter','latex','FontSize',18);
q(1).EdgeColor = 'none';
q(2).EdgeColor = 'none';
q(3).EdgeColor = 'none';
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
zlabel('$\it{z}$','interpreter','latex','FontSize',14);
colorbar;
print('InviscidBurgers_MPI_CUDA_3d','-dpng');

%% Load and plot Initial Condition
fID = fopen('initial.bin');
output = fread(fID,[1,nx*ny*nz],'float')';
u0 = reshape(output,nx,ny,nz);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(2)
q=slice(x,y,z,u0,0,0,0); axis(region); colormap(getColorMap('kwave')); view(90,90);
title('Initial Condition','interpreter','latex','FontSize',18);
q(1).EdgeColor = 'none';
q(2).EdgeColor = 'none';
q(3).EdgeColor = 'none';
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
zlabel('$\it{z}$','interpreter','latex','FontSize',14);
colorbar;
print('InitialCondition_MPI_CUDA_3d','-dpng');

% Clean up
! rm -rf *.bin 
