%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Solving 2-D heat equation with jacobi method
%
%                   u_t = D*(u_xx + u_yy) + s(u), 
%         for (x,y) \in [0,L]x[0,W] and S = s(u): source term
%
%             coded by Manuel Diaz, manuel.ade'at'gmail.com
%        National Health Research Institutes, NHRI, 2016.02.11
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; %close all; clc;

%% Parameters
D = 1.0; % alpha
t0 = 0.1; % intial time
tFinal = 1.0;	% End time
L = 10; nx = 32; dx = L/(nx-1); 
W = 10; ny = 32; dy = W/(ny-1);
Dx = D/dx^2; Dy = D/dy^2; 

% Build Numerical Mesh
[x,y] = meshgrid(-L/2:dx:L/2,-W/2:dy:W/2);

% Add source term
sourcefun='dont'; % add source term
switch sourcefun
    case 'add';     S = @(w) 0.1*w.^2;
    case 'dont';    S = @(w) zeros(size(w));
end

% Build IC
d=1; u0 = exp( -(x.^2 + y.^2)/(4*d*t0) );

% Build Exact solution
uE = t0/tFinal*exp(-(x.^2 + y.^2)/(4*d*tFinal) );

% Set Initial time step
dt0 = 1/(2*D*(1/dx^2+1/dy^2))*0.9; % stability condition

% Set plot region
region = [-L/2,L/2,-W/2,W/2,0,1]; 

%% Solver Loop 
% load initial conditions 
t=t0; it=0; u=u0; dt=dt0;
 
while t < tFinal
    % RK stages
    uo=u;
     
    % 1st stage
    dF=Laplace2d(u,nx,ny,Dx,Dy);
    u=uo+dt*dF; 
    
    % 2nd Stage
    dF=Laplace2d(u,nx,ny,Dx,Dy);
    u=0.75*uo+0.25*(u+dt*dF); 

    % 3rd stage
    dF=Laplace2d(u,nx,ny,Dx,Dy);
    u=(uo+2*(u+dt*dF))/3; 
    
    % set BCs
    u(1,:) = 0; u(nx,:) = 0;
    u(:,1) = 0; u(:,ny) = 0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
    % plot solution
    if mod(it,100); surf(x,y,u); axis(region); drawnow; end
end
 
%% % Post Process 
% Final Plot
subplot(121); h=surf(x,y,u); axis(region);
title('heat2d, Cell Averages','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
subplot(122); q=surf(x,y,uE); axis(region);
title('heat2d, Exact solution','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
print('diffusion2d','-dpng');

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*dy*sum(abs(err)); fprintf('L_1 norm: %1.2e \n',L1);
L2 = (dx*dy*sum(err.^2))^0.5; fprintf('L_2 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);