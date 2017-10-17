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
D = 1.0; % Alpha
t0 = 0.1;	% Initial time
tFinal = 0.5;	% End time
L = 10; nx = 101; dx = L/(nx-1); 
Dx = D/dx^2; 

% Build Numerical Mesh
x = -L/2:dx:L/2; 

% Add source term
sourcefun='dont'; % add source term
switch sourcefun
    case 'add';     S = @(w) 0.1*w.^2;
    case 'dont';    S = @(w) zeros(size(w));
end

% Build IC
d=1; u0 = exp( -x.^2/(4*d*t0) );

% Build Exact solution
uE = sqrt(t0/tFinal)*exp( -x.^2/(4*d*tFinal) );

% Set Initial time step
dt0 = 1/(2*D*(1/dx^2))*0.9; % stability condition

% Set plot region
region = [-L/2,L/2,0,1]; 

%% Solver Loop 
% load initial conditions 
t=0.1; it=0; u=u0; dt=dt0;
 
while t < tFinal
    
    % RK stages
    uo=u;
     
    % 1st stage
    dF=Laplace1d(u,nx,Dx);
    u=uo+dt*dF; 
    
    % 2nd Stage
    dF=Laplace1d(u,nx,Dx);
    u=0.75*uo+0.25*(u+dt*dF); 

    % 3rd stage
    dF=Laplace1d(u,nx,Dx);
    u=(uo+2*(u+dt*dF))/3; 
    
    % set BCs
    u(1)=0; u( nx )=0;
    u(2)=0; u(nx-1)=0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
    % plot solution
    if mod(it,100); plot(x,u,'.b'); axis(region); drawnow; end
end
 
%% % Post Process 
% Final Plot
h=plot(x,u,'.b',x,uE,'-r',x,u0,'--k'); axis(region); grid on;
title('heat1d, Cell Averages','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{u(x)}$','interpreter','latex','FontSize',14);
legend('Jacobi Method','Exact Solution','IC');
print('diffusion1d','-dpng');

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*sum(abs(err)); fprintf('L_1 norm: %1.2e \n',L1);
L2 = (dx*sum(err.^2))^0.5; fprintf('L_2 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);