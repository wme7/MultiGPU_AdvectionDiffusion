%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Solving 1-D wave equation with 5th order
%          Weighted Essentially Non-Oscilaroty (MOL-WENO5-LF)
%
%                 du/dt + df/dx = S, for x \in [a,b]
%                  where f = f(u): linear/nonlinear
%                     and S = s(u): source term
%
%             coded by Manuel Diaz, manuel.ade'at'gmail.com 
%            Institute of Applied Mechanics, NTU, 2012.08.20
%                               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref: C.-W. Shu, High order weighted essentially non-oscillatory schemes
% for convection dominated problems, SIAM Review, 51:82-126, (2009). 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes: A fully conservative finite volume implementation of the method of
% lines (MOL) using WENO5 associated with SSP-RK33 time integration method. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; %close all; clc;

%% Parameters
   nx = 0101;	% number of cells
   ny = 0101;	% number of cells
   nz = 0101;	% number of cells
  CFL = 0.40;	% Courant Number
 tEnd = 0.40;	% End time
 
fluxfun='burgers'; % select flux function
% Define our Flux function
switch fluxfun
    case 'linear'   % Scalar Advection, CFL_max: 0.65
        c=-1; flux = @(w) c*w; 
        dflux = @(w) c*ones(size(w));
    case 'burgers' % Burgers, CFL_max: 0.40  
        flux = @(w) w.^2/2; 
        dflux = @(w) w; 
    case 'buckley' % Buckley-Leverett, CFL_max: 0.20 & tEnd: 0.40
        flux = @(w) 4*w.^2./(4*w.^2+(1-w).^2);
        dflux = @(w) 8*w.*(1-w)./(5*w.^2-2*w+1).^2;
end

sourcefun='dont'; % add source term
% Source term
switch sourcefun
    case 'add'
        S = @(w) 0.1*w.^2;
    case 'dont'
        S = @(w) zeros(size(w));
end

% Build discrete domain
ax=-1; bx=1; dx=(bx-ax)/(nx-1); xc=ax:dx:bx;
ay=-1; by=1; dy=(by-ay)/(ny-1); yc=ay:dy:by;
az=-1; bz=1; dz=(bz-az)/(nz-1); zc=az:dz:bz;
[x,y,z]=meshgrid(xc,yc,zc);

% Build IC
u0=1.0*exp(-(x.^2+y.^2+z.^2)/0.1);

% Set plot region
region = [ax,bx,ay,by,az,bz]; 

%% Solver Loop

% load initial conditions
t=0; it=0; u=u0; 

while t < tEnd
    
	% Update/correct time step
    dt=CFL*dx/max(abs(u(:))); if t+dt>tEnd, dt=tEnd-t; end
    
	% Update time and iteration counter
    t=t+dt; it=it+1;
    
    % RK Initial step
    uo=u;
    
    % 1st stage
    dF=WENO5resAdv_X(u,flux,dflux,dx,nx); 
    dG=WENO5resAdv_Y(u,flux,dflux,dy,ny);
    dH=WENO5resAdv_Z(u,flux,dflux,dz,nz);
    u=uo-dt*(dF+dG+dH);
    
    % 2nd Stage
    dF=WENO5resAdv_X(u,flux,dflux,dx,nx);
    dG=WENO5resAdv_Y(u,flux,dflux,dy,ny);
    dH=WENO5resAdv_Z(u,flux,dflux,dz,nz);
    u=0.75*uo+0.25*(u-dt*(dF+dG+dH));

    % 3rd stage
    dF=WENO5resAdv_X(u,flux,dflux,dx,nx);
    dG=WENO5resAdv_Y(u,flux,dflux,dy,ny);
    dH=WENO5resAdv_Z(u,flux,dflux,dz,nz);
    u=(uo+2*(u-dt*(dF+dG+dH)))/3;

end

%% Post-process
q=slice(x,y,z,u,bx/3,by/3,bz/3); axis(region);
title('Inviscid Burgers, Matlab-WENO5-RK3','interpreter','latex','FontSize',18);
q(1).EdgeColor = 'none';
q(2).EdgeColor = 'none';
q(3).EdgeColor = 'none';
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
zlabel('$\it{z}$','interpreter','latex','FontSize',14);
colorbar;
print('InviscidBurgers_WENO5_3d','-dpng');