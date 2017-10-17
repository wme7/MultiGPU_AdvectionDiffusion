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
% NOTE: Exact solution taken from: 
%            http://eqworld.ipmnet.ru/en/solutions/lpde/lpde104.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; %close all; clc;

%% Parameters
D = 0.27; % alpha
t0 = 1.0; % intial time
tFinal = 2.5;	% End time
L = 10; nr = 128; dr = L/(nr-1); 
W = 10; ny = 32; dy = W/(ny-1);
Dr = D/(12*dr^2); Dy = D/(12*dy^2); 

% Build Numerical Mesh
[r,y] = meshgrid(-L/2:dr:L/2,-W/2:dy:W/2);
inv_r = 1./(-L/2:dr:L/2); inv_r(inv_r==inf)=0; % dr/r

% Add source term
sourcefun='dont'; % add source term
switch sourcefun
    case 'add';     S = @(w) 0.1*w.^2;
    case 'dont';    S = @(w) zeros(size(w));
end

% Build IC
d=D; u0 = exp( -(r.^2)/(4*d*t0) );

% Build Exact solution
uE = 1/tFinal*exp(-(r.^2)/(4*d*tFinal) );

% Set Initial time step
dt0 = 1/(2*D*(1/dr^2+1/dy^2))*0.9; % stability condition

% Set plot region
region = [-L/2,L/2,-W/2,W/2,0,1]; 

%% Solver Loop 
% load initial conditions 
t=t0; it=0; u=u0; dt=dt0;
 
while t < tFinal
    % RK stages
    uo=u;
     
    % 1st stage
    dF=Laplace2d_axisymmetric(u,nr,ny,Dr,Dy,dr,inv_r);
    u=uo+dt*dF; % or:
    %dF=Laplace2d(u,nr,ny,Dr,Dy);
    %S =RadCorr2d(u,nr,ny,Dr,dr,inv_r);
    %u=uo+dt*(dF+S); 
    
    % set BCs
    u(1,:) = u(3,:); u( ny ,:) = u(ny-2,:);
    u(2,:) = u(3,:); u(ny-1,:) = u(ny-2,:);
    u(:,1) = 0; u(:,nr) = 0;
    
    % 2nd Stage
    dF=Laplace2d_axisymmetric(u,nr,ny,Dr,Dy,dr,inv_r);
    u=0.75*uo+0.25*(u+dt*dF); % or:
    %dF=Laplace2d(u,nr,ny,Dr,Dy);
    %S =RadCorr2d(u,nr,ny,Dr,dr,inv_r);
    %u=0.75*uo+0.25*(u+dt*(dF+S)); 

    % set BCs
    u(1,:) = u(3,:); u( ny ,:) = u(ny-2,:);
    u(2,:) = u(3,:); u(ny-1,:) = u(ny-2,:);
    u(:,1) = 0; u(:,nr) = 0;
    
    % 3rd stage
    dF=Laplace2d_axisymmetric(u,nr,ny,Dr,Dy,dr,inv_r);
    u=(uo+2*(u+dt*dF))/3; % or:
    %dF=Laplace2d(u,nr,ny,Dr,Dy);
    %S =RadCorr2d(u,nr,ny,Dr,dr,inv_r);
    %u=(uo+2*(u+dt*(dF+S)))/3; 
    
    % set BCs
    u(1,:) = u(3,:); u( ny ,:) = u(ny-2,:);
    u(2,:) = u(3,:); u(ny-1,:) = u(ny-2,:);
    u(:,1) = 0; u(:,nr) = 0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
    % plot solution
    if mod(it,100); surf(r,y,u); axis(region); drawnow; end
end
 
%% % Post Process 
% Final Plot
plot(r,u(ny/2,:),'.r',r,uE(ny/2,:),'-k',r,u0(ny/2,:),'--k');
xlabel('$\it{r}$','interpreter','latex','FontSize',14);
ylabel('$\it{T}$','interpreter','latex','FontSize',14);
print('diffusion2d_axisymmetric','-dpng');

% Error norms
err = abs(uE(:)-u(:));
L1 = dr*dy*sum(abs(err)); fprintf('L_1 norm: %1.2e \n',L1);
L2 = (dr*dy*sum(err.^2))^0.5; fprintf('L_2 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);