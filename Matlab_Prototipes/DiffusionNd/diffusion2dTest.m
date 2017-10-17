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
function [L1,Linf] = diffusion2dTest(nx,ny,tFinal,param)
%% Parameters
D = 1.0; % alpha
t0 = 0.1; % Initial time
L = 10; dx = L/(nx-1); 
W = 10; dy = W/(ny-1);
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
d=1; u0 = exp( -(x.^2+y.^2)/(4*d*t0) );

% Build Exact solution
uE = t0/tFinal*exp( -(x.^2+y.^2)/(4*d*tFinal) );

% Set Initial time step
dt0 = 1/(2*D*(1/dx^2+1/dy^2))*param; % stability condition

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
    
end

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*dy*sum(abs(err)); fprintf('L_1 norm: %1.2e \n',L1);
L2 = (dx*dy*sum(err.^2))^0.5; fprintf('L_2 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);