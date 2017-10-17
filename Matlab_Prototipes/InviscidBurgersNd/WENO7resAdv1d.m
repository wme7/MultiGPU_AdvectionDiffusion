function res = WENO7resAdv1d(w,flux,dflux,dx,N)
% *************************************************************************
% Input: u(i) = [u(i-2) u(i-1) u(i) u(i+1) u(i+2)];
% Output: res = df/dx;
%
% Based on:
% C.W. Shu's Lectures notes on: 'ENO and WENO schemes for Hyperbolic
% Conservation Laws' 
%
% coded by Manuel Diaz, 02.10.2012, NTU Taiwan.
% *************************************************************************
%
% Domain cells (I{i}) reference:
%
%                |           |   u(i)    |           |
%                |  u(i-1)   |___________|           |
%                |___________|           |   u(i+1)  |
%                |           |           |___________|
%             ...|-----0-----|-----0-----|-----0-----|...
%                |    i-1    |     i     |    i+1    |
%                |-         +|-         +|-         +|
%              i-3/2       i-1/2       i+1/2       i+3/2
%
% ENO stencils (S{r}) reference:
%
%                               |_______________S3______________|
%                               |                               |
%                       |______________S2_______________|       |
%                       |                               |       |
%               |______________S1_______________|       |       |
%               |                               |       |       |
%       |_______________S0______________|       |       |       |
%     ..|---o---|---o---|---o---|---o---|---o---|---o---|---o---|...
%       | I{i-3}| I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}| I{i+3}|
%                                      -|
%                                     i+1/2
%
%       |______________S0_______________|
%       |                               |
%       |       |______________S1_______________|
%       |       |                               |
%       |       |       |______________S2_______________|
%       |       |       |                               |
%       |       |       |       |_______________S3______________|
%     ..|---o---|---o---|---o---|---o---|---o---|---o---|---o---|...
%       | I{i-3}| I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|| I{i+3}
%                               |+
%                             i-1/2
%
% WENO stencil: S{i} = [ I{i-3},...,I{i+3} ]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The stencil size
I=4:(N+3); 

% Allocate ghost cells and solution arrays
W=[w(:,1),w(:,1),w(:,1),w(:,I-3),w(:,N),w(:,N),w(:,N),w(:,N)];
hn=zeros(size(W));
hp=zeros(size(W));

% Lax-Friedrichs Flux Splitting
a=max(abs(dflux(W))); 
V=0.5*(flux(W)+a.*W); 
U=0.5*(flux(W)-a.*W);

%% Right Flux: f_{i+1/2}^{-}
vmmm= V(:,I-3);
vmm = V(:,I-2);
vm  = V(:,I-1);
vo  = V(:, I );
vp  = V(:,I+1);
vpp = V(:,I+2);
vppp= V(:,I+3);

% Smooth Indicators
B0n = vm.*(134241*vm-114894*vo)   +vmmm.*(56694*vm-47214*vmm+6649*vmmm-22778*vo)...
        +25729*vo.^2  +vmm.*(-210282*vm+85641*vmm+86214*vo);
B1n = vo.*(41001*vo-30414*vp)     +vmm.*(-19374*vm+3169*vmm+19014*vo-5978*vp)...
        +6649*vp.^2   +vm.*(33441*vm-70602*vo+23094*vp);
B2n = vp.*(33441*vp-19374*vpp)    +vm.*(6649*vm-30414*vo+23094*vp-5978*vpp)...
        +3169*vpp.^2  +vo.*(41001*vo-70602*vp+19014*vpp);
B3n = vpp.*(85641*vpp-47214*vppp) +vo.*(25729*vo-114894*vp+86214*vpp-22778*vppp)...
        +6649*vppp.^2 +vp.*(134241*vp-210282*vpp+56694*vppp);

% Constants
g0 = 1/35; g1 = 12/35; g2 = 18/35; g3 = 4/35; epsilon = 1e-6;

% Alpha weights
alpha0n = g0./(epsilon + B0n).^2;
alpha1n = g1./(epsilon + B1n).^2;
alpha2n = g2./(epsilon + B2n).^2;
alpha3n = g3./(epsilon + B3n).^2;
alphasumn = alpha0n + alpha1n + alpha2n + alpha3n;

% Non-linear weigths
w0n = alpha0n./alphasumn;
w1n = alpha1n./alphasumn;
w2n = alpha2n./alphasumn;
w3n = alpha3n./alphasumn;

% Numerical Flux at cell boundary, $u_{i+1/2}^{-}$;
hn(:,I) = w0n.*(-3*vmmm + 13*vmm - 23*vm  + 25*vo  )/12 + ...
        w1n.*( 1*vmm  - 5*vm   + 13*vo  +  3*vp  )/12 + ...
        w2n.*(-1*vm   + 7*vo   +  7*vp  -  1*vpp )/12 + ...
        w3n.*( 3*vo   + 13*vp  -  5*vpp +  1*vppp)/12;

%% Left Flux: f_{i+1/2}^{+}
ummm= U(:,I-2);
umm = U(:,I-1);
um  = U(:, I );
uo  = U(:,I+1);
up  = U(:,I+2);
upp = U(:,I+3);
uppp= U(:,I+4);

% Smooth Indicators
B0p = um.*(134241*um-114894*uo)   +ummm.*(56694*um-47214*umm+6649*ummm-22778*uo)...
        +25729*uo.^2  +umm.*(-210282*um+85641*umm+86214*uo);
B1p = uo.*(41001*uo-30414*up)     +umm.*(-19374*um+3169*umm+19014*uo-5978*up)...
        +6649*up.^2   +um.*(33441*um-70602*uo+23094*up);
B2p = up.*(33441*up-19374*upp)    +um.*(6649*um-30414*uo+23094*up-5978*upp)...
        +3169*upp.^2  +uo.*(41001*uo-70602*up+19014*upp);
B3p = upp.*(85641*upp-47214*uppp) +uo.*(25729*uo-114894*up+86214*upp-22778*uppp)...
        +6649*uppp.^2 +up.*(134241*up-210282*upp+56694*uppp);

% Constants
g0 = 4/35; g1 = 18/35; g2 = 12/35; g3 = 1/35; epsilon = 1e-6;

% Alpha weights
alpha0p = g0./(epsilon + B0p).^2;
alpha1p = g1./(epsilon + B1p).^2;
alpha2p = g2./(epsilon + B2p).^2;
alpha3p = g3./(epsilon + B3p).^2;
alphasump = alpha0p + alpha1p + alpha2p + alpha3p;

% Non-linear weigths
w0p = alpha0p./alphasump;
w1p = alpha1p./alphasump;
w2p = alpha2p./alphasump;
w3p = alpha3p./alphasump;

% Numerical Flux at cell boundary, $u_{i+1/2}^{+}$;
hp(:,I) = w0p.*( 1*ummm - 5*umm  + 13*um  +  3*uo  )/12 + ...
        w1p.*(-1*umm  + 7*um   +  7*uo  -  1*up  )/12 + ... 
        w2p.*( 3*um   + 13*uo  -  5*up  +  1*upp )/12 + ...
        w3p.*(25*uo   - 23*up  + 13*upp -  3*uppp)/12;

%% Compute finite difference residual term, df/dx.
res = (hp(:,I)+hn(:,I)-hp(:,I-1)-hn(:,I-1))/dx;