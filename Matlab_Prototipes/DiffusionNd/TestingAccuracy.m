%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                  Testing the Order Of Accuracy
%
%              coded by Manuel Diaz, NTU, 2016.05.09
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

% Fixed Parameters
tEnd = 0.5; % One cycle for every test
factor = 0.9; % stability factor

% Parameters
mth = [1,2,3]; % solver: {1}1d, {2}2d, {3}3d.
nc  = [9,17,33,65]; % number of cells to use in every test.

% Number of parameters
p1 = length(mth);
p2 = length(nc);

% Allocate space for results
table = zeros(p2,2,p1,1);
Norm = zeros(size(table));
OOA = zeros(size(table));

%% Compute L1 and L\infty norms

for n = 1:p2
    tic
    [Norm(n,1,1),Norm(n,2,1)] = diffusion1dTest(nc(n),tEnd,factor);
    toc
    tic
    [Norm(n,1,2),Norm(n,2,2)] = diffusion2dTest(nc(n),nc(n),tEnd,factor);
    toc
    tic
    [Norm(n,1,3),Norm(n,2,3)] = diffusion3dTest(nc(n),nc(n),nc(n),tEnd,factor);
    toc
end

%% Compute the Order of Accuracy (OOA)

for l = 1:p1
    for n = 2:p2
        OOA(n,1,l) = log(Norm(n-1,1,l)/Norm(n,1,l))/log(2);
        OOA(n,2,l) = log(Norm(n-1,2,l)/Norm(n,2,l))/log(2);
    end
end

%% Plot figure with results
loglog(nc,Norm(:,:,1),'-s',...
    nc,Norm(:,:,2),'-o',...
    nc,Norm(:,:,3),'-h');
legend(...
    'heat1d L1','heat1d Linf',...
    'heat2d L1','heat2d Linf',...
    'heat3d L1','heat3d Linf');

%% Display Result
for l = 1:p1
    fprintf('***************************************************************\n')
    fprintf(' Heat solver %d D\n',mth(l));
    fprintf('***************************************************************\n')
    fprintf(' nE \t L1-Norm \t Degree \t Linf-Norm \t Degree\n');
    for n = 1:p2
        fprintf('%3.0f \t %1.2e \t %2.2f \t\t %1.2e \t %2.2f \n',...
        nc(n)-1,Norm(n,1,l),OOA(n,1,l),Norm(n,2,l),OOA(n,2,l));
    end
end
fprintf('\n');

% Manuel Diaz, NHRI, 2016
% End of Test