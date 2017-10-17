
/***********************/
/* WENO RECONSTRUCTION */
/***********************/
__device__ REAL WENO5reconstruction(
  const REAL vmm,
  const REAL vm,
  const REAL v,
  const REAL vp,
  const REAL vpp,
  const REAL umm,
  const REAL um,
  const REAL u,
  const REAL up,
  const REAL upp)
{
  REAL B0, B1, B2, a0, a1, a2, alphasum, dflux;
  
  // Smooth Indicators (Beta factors)
  B0 = C1312*(vmm-2*vm+v  )*(vmm-2*vm+v  ) + C14*(vmm-4*vm+3*v)*(vmm-4*vm+3*v);
  B1 = C1312*(vm -2*v +vp )*(vm -2*v +vp ) + C14*(vm-vp)*(vm-vp);
  B2 = C1312*(v  -2*vp+vpp)*(v  -2*vp+vpp) + C14*(3*v-4*vp+vpp)*(3*v-4*vp+vpp);
  
  // Alpha weights
  a0 = D0N/((EPS + B0)*(EPS + B0));
  a1 = D1N/((EPS + B1)*(EPS + B1));
  a2 = D2N/((EPS + B2)*(EPS + B2));
  alphasum = a0 + a1 + a2;
  
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  dflux =(a0*(2*vmm- 7*vm + 11*v) +
          a1*( -vm + 5*v  + 2*vp) +
          a2*( 2*v + 5*vp - vpp ))/(6*alphasum);

  // Smooth Indicators (Beta factors)
  B0 = C1312*(umm-2*um+u  )*(umm-2*um +u  ) + C14*(umm-4*um+3*u)*(umm-4*um+3*u);
  B1 = C1312*(um -2*u +up )*(um -2*u  +up ) + C14*(um-up)*(um-up);
  B2 = C1312*(u  -2*up+upp)*(u  -2*up +upp) + C14*(3*u-4*up+upp)*(3*u-4*up+upp);
  
  // Alpha weights
  a0 = D0P/((EPS + B0)*(EPS + B0));
  a1 = D1P/((EPS + B1)*(EPS + B1));
  a2 = D2P/((EPS + B2)*(EPS + B2));
  alphasum = a0 + a1 + a2;

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  dflux+=(a0*( -umm + 5*um + 2*u  ) +
          a1*( 2*um + 5*u  - up   ) +
          a2*(11*u  - 7*up + 2*upp))/(6*alphasum);
  
  // Compute the numerical flux v_{i+1/2}
  return dflux;
}

__device__ REAL WENO5Zreconstruction(
  const REAL vmm,
  const REAL vm,
  const REAL v,
  const REAL vp,
  const REAL vpp,
  const REAL umm,
  const REAL um,
  const REAL u,
  const REAL up,
  const REAL upp)
{
  REAL B0, B1, B2, a0, a1, a2, tau5, alphasum, dflux;
  
  // Smooth Indicators (Beta factors)
  B0 = C1312*(vmm-2*vm+v  )*(vmm-2*vm+v  ) + C14*(vmm-4*vm+3*v)*(vmm-4*vm+3*v);
  B1 = C1312*(vm -2*v +vp )*(vm -2*v +vp ) + C14*(vm-vp)*(vm-vp);
  B2 = C1312*(v  -2*vp+vpp)*(v  -2*vp+vpp) + C14*(3*v-4*vp+vpp)*(3*v-4*vp+vpp);
  
  // Alpha weights
  tau5 = fabs(B0-B2);
  a0 = D0N*(1.+tau5/(B0+EPS));
  a1 = D1N*(1.+tau5/(B1+EPS));
  a2 = D2N*(1.+tau5/(B2+EPS));
  alphasum = a0 + a1 + a2;
  
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  dflux =(a0*(2*vmm- 7*vm + 11*v) +
          a1*( -vm + 5*v  + 2*vp) +
          a2*( 2*v + 5*vp - vpp ))/(6*alphasum);

  // Smooth Indicators (Beta factors)
  B0 = C1312*(umm-2*um+u  )*(umm-2*um +u  ) + C14*(umm-4*um+3*u)*(umm-4*um+3*u);
  B1 = C1312*(um -2*u +up )*(um -2*u  +up ) + C14*(um-up)*(um-up);
  B2 = C1312*(u  -2*up+upp)*(u  -2*up +upp) + C14*(3*u-4*up+upp)*(3*u-4*up+upp);
  
  // Alpha weights
  tau5 = fabs(B0-B2);
  a0 = D0P*(1.+tau5/(B0+EPS));
  a1 = D1P*(1.+tau5/(B1+EPS));
  a2 = D2P*(1.+tau5/(B2+EPS));
  alphasum = a0 + a1 + a2;

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  dflux+=(a0*( -umm + 5*um + 2*u  ) +
          a1*( 2*um + 5*u  - up   ) +
          a2*(11*u  - 7*up + 2*upp))/(6*alphasum);
  
  // Compute the numerical flux v_{i+1/2}
  return dflux;
}

__global__ void Compute_dH(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu,
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz, 
  const REAL dz)
{
  // Temporary variables
  REAL fu, fu_old;
  REAL f1mm, f1m, f1, f1p, f1pp;
  REAL g1mm, g1m, g1, g1p, g1pp;

  // Indexes
  unsigned int i, j, k, xy=pitch*ny;

  // local threads indexes
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;

  // Compute only for internal nodes
  if (i>2 && i<nx-3 && j>2 && j<ny-3) {

    // Old resulst arrays
    fu_old=0;
    
    f1mm= 0.5*( Flux(u[i+pitch*j+xy*0]) + fabs(u[i+pitch*j+xy*0])*u[i+pitch*j+xy*0]); // node(i-2)
    f1m = 0.5*( Flux(u[i+pitch*j+xy*1]) + fabs(u[i+pitch*j+xy*1])*u[i+pitch*j+xy*1]); // node(i-1)
    f1  = 0.5*( Flux(u[i+pitch*j+xy*2]) + fabs(u[i+pitch*j+xy*2])*u[i+pitch*j+xy*2]); // node( i )     imm--im--i--ip--ipp--ippp
    f1p = 0.5*( Flux(u[i+pitch*j+xy*3]) + fabs(u[i+pitch*j+xy*3])*u[i+pitch*j+xy*3]); // node(i+1)
       
    g1mm= 0.5*( Flux(u[i+pitch*j+xy*1]) - fabs(u[i+pitch*j+xy*1])*u[i+pitch*j+xy*1]); // node(i-1)
    g1m = 0.5*( Flux(u[i+pitch*j+xy*2]) - fabs(u[i+pitch*j+xy*2])*u[i+pitch*j+xy*2]); // node( i )     imm--im--i--ip--ipp--ippp
    g1  = 0.5*( Flux(u[i+pitch*j+xy*3]) - fabs(u[i+pitch*j+xy*3])*u[i+pitch*j+xy*3]); // node(i+1)
    g1p = 0.5*( Flux(u[i+pitch*j+xy*4]) - fabs(u[i+pitch*j+xy*4])*u[i+pitch*j+xy*4]); // node(i+2)
        
    for (k = 2; k < nz-3; k++) {
        
      // Compute and split fluxes
      f1pp= 0.5*( Flux(u[i+pitch*j+xy*(k+2)]) + fabs(u[i+pitch*j+xy*(k+2)])*u[i+pitch*j+xy*(k+2)]); // node(i+2)
      g1pp= 0.5*( Flux(u[i+pitch*j+xy*(k+3)]) - fabs(u[i+pitch*j+xy*(k+3)])*u[i+pitch*j+xy*(k+3)]); // node(i+3)
      
      // Reconstruct
      fu = WENO5Zreconstruction(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);
      
      // Compute Lq = dH/dz
      Lu[i+pitch*j+xy*k]-=(fu-fu_old)/dz; // dudz
      
      // Save old results
      fu_old=fu;
      
      f1mm= f1m;   // node(i-2)
      f1m = f1;    // node(i-1)
      f1  = f1p;   // node( i )    imm--im--i--ip--ipp--ippp
      f1p = f1pp;  // node(i+1)
      
      g1mm= g1m;   // node(i-1)
      g1m = g1;    // node( i )    imm--im--i--ip--ipp--ippp
      g1  = g1p;   // node(i+1)
      g1p = g1pp;  // node(i+2)
    }
  }
}
