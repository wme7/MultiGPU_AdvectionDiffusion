% Produce Run

% =========================== HeatEq2D (13-pt) =======================
% Optimization                                 :  none
% Kernel time ex. data transfers               :  11.430000 seconds
% ===================================================================
% Total effective GFLOPs                       :  7.013130
% ===================================================================
% 2D Grid Size                                 :  1001 x 1001 
% Iterations                                   :  10000
% Final Time                                   :  0.325
% ===================================================================

% Set run parameters
C = 1;
L = 10;
W = 10;
nx = 1001;
ny = 1001;
iter = 10000;
block_X = 16;
block_Y = 16;

% Write sh.run
fID = fopen('run.sh','wt');
fprintf(fID,'make\n');
args = sprintf('%1.1f %1.2f %1.2f %d %d %d %d %d',C,L,W,nx,ny,iter,block_X,block_Y);
fprintf(fID,'./diffusion2d.run %s\n',args);
fclose(fID);

% Execute sh.run
! sh run.sh

% Load data
fID = fopen('result.bin');
output = fread(fID,[1,nx*ny],'float')';

% Plot data
myplot(output,nx,ny,L,W);

% Clean up
! rm -rf *.bin
