//
//  kernels.cu
//  Burgers3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

extern "C" {
#include "Burgers.h"
}

/*******************************/
/* Define Textures & Constanst */
/*******************************/
__constant__ REAL d_kx;
__constant__ REAL d_ky;
__constant__ REAL d_kz;
texture<int2, 2, cudaReadModeElementType> tex_u;

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

/*********************************************/
/* A method for checking error in CUDA calls */
/*********************************************/
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
  #if defined(DISPL)
    if (error != cudaSuccess)
    {
    printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
    exit(-1);
    }
  #endif
  return;
}

/*****************/
/* FLUX FUNCTION */
/*****************/
__device__ REAL Flux(
  const REAL u){
  return 0.5*u*u;
}

// *************************************************************************
// Input: v(i) = [v(i-2) v(i-1) v(i) v(i+1) v(i+2) v(i+3)];
// Output: res = df/dx;
//
// Based on:
// C.W. Shu's Lectures notes on: 'ENO and WENO schemes for Hyperbolic
// Conservation Laws'
//
// coded by Manuel Diaz, 02.10.2012, NTU Taiwan.
// *************************************************************************
//
// Domain cells (I{i}) reference:
//
//                |           |   u(i)    |           |
//                |  u(i-1)   |___________|           |
//                |___________|           |   u(i+1)  |
//                |           |           |___________|
//             ...|-----0-----|-----0-----|-----0-----|...
//                |    i-1    |     i     |    i+1    |
//                |-         +|-         +|-         +|
//              i-3/2       i-1/2       i+1/2       i+3/2
//
// ENO stencils (S{r}) reference:
//
//                           |___________S2__________|
//                           |                       |
//                   |___________S1__________|       |
//                   |                       |       |    using only f^{+}
//           |___________S0__________|       |       |
//         ..|---o---|---o---|---o---|---o---|---o---|...
//           | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
//                                  -|
//                                 i+1/2
//
//                   |___________S0__________|
//                   |                       |
//                   |       |___________S1__________|    using only f^{-}
//                   |       |                       |
//                   |       |       |___________S2__________|
//                 ..|---o---|---o---|---o---|---o---|---o---|...
//                   | I{i-1}|  I{i} | I{i+1}| I{i+2}| I{i+3}|
//                                   |+
//                                 i+1/2
//
// WENO stencil: S{i} = [ I{i-2},...,I{i+3} ]
// *************************************************************************

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

/*****************/
/* Compute du/dx */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
__global__ void Compute_dF(
  REAL * __restrict__ Lu, 
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny, 
  const REAL dx,
  const int k)
{
  // Shared variables
  __shared__ REAL sfu[WIDTH][TILE+1]; // faces = inner nodes + 1

  // Temporary variables
  REAL u0,u1,u2,u3,u4,u5;
  REAL B0,B1,B2,a0,a1,a2;
  REAL umm,um,uo,up,upp,tau5,asum;

  // Indexes
  unsigned int i,j,si,sj,o; int2 Data;
  
  // Global threads indexes
  i =    TILE    * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;

  // shared memory indexes
  si = threadIdx.x;
  sj = threadIdx.y;

  // Global index
  o = i+pitch*j+pitch*ny*k;

  if (i < nx){
    // Load data from texture memory
    Data = tex2D(tex_u,i-3,j); u0 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i-2,j); u1 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i-1,j); u2 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u, i ,j); u3 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i+1,j); u4 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i+2,j); u5 = __hiloint2double(Data.y, Data.x);

    // Reconstruct Face fluxes
    // Split data for f_{i}^{+}
    umm=0.5*(Flux(u0) + fabs(u0)*u0);
    um =0.5*(Flux(u1) + fabs(u1)*u1);
    uo =0.5*(Flux(u2) + fabs(u2)*u2);
    up =0.5*(Flux(u3) + fabs(u3)*u3);
    upp=0.5*(Flux(u4) + fabs(u4)*u4);
    
    // Smooth Indicators (Beta factors)
    B0 = C1312*(umm-2*um+uo )*(umm-2*um +uo ) + 0.25*(umm-4*um+3*uo)*(umm-4*um+3*uo);
    B1 = C1312*(um -2*uo+up )*(um -2*uo +up ) + 0.25*(um-up)*(um-up);
    B2 = C1312*(uo -2*up+upp)*(uo -2*up +upp) + 0.25*(3*uo-4*up+upp)*(3*uo-4*up+upp);
      
    // Alpha weights
    tau5 = fabs(B0-B2);
    a0 = D0N*(1.+tau5/(B0+EPS));
    a1 = D1N*(1.+tau5/(B1+EPS));
    a2 = D2N*(1.+tau5/(B2+EPS));
    asum = a0 + a1 + a2;
    
    // Numerical Flux at cell boundary, $u_{i+1/2}^{-}$;
    sfu[sj][si] = (
      a0*(2*umm- 7*um + 11*uo) +
      a1*( -um + 5*uo + 2*up ) +
      a2*( 2*uo+ 5*up - upp  ))/(6*asum);

    // split data for f_{i}^{-}
    umm=0.5*(Flux(u1) - fabs(u1)*u1);
    um =0.5*(Flux(u2) - fabs(u2)*u2);
    uo =0.5*(Flux(u3) - fabs(u3)*u3);
    up =0.5*(Flux(u4) - fabs(u4)*u4);
    upp=0.5*(Flux(u5) - fabs(u5)*u5);

    // Smooth Indicators (Beta factors)
    B0 = C1312*(umm-2*um+uo )*(umm-2*um +uo ) + 0.25*(umm-4*um+3*uo)*(umm-4*um+3*uo);
    B1 = C1312*(um -2*uo+up )*(um -2*uo +up ) + 0.25*(um-up)*(um-up);
    B2 = C1312*(uo -2*up+upp)*(uo -2*up +upp) + 0.25*(3*uo-4*up+upp)*(3*uo-4*up+upp);
    
    // Alpha weights
    tau5 = fabs(B0-B2);
    a0 = D0P*(1.+tau5/(B0+EPS));
    a1 = D1P*(1.+tau5/(B1+EPS));
    a2 = D2P*(1.+tau5/(B2+EPS));
    asum = a0 + a1 + a2;

    // Numerical Flux at cell boundary, $u_{i+1/2}^{+}$;
    sfu[sj][si] += (
      a0*( -umm + 5*um + 2*uo ) +
      a1*( 2*um + 5*uo - up   ) +
      a2*(11*uo - 7*up + 2*upp))/(6*asum);
    __syncthreads();

    // Compute Lq = (f_{i+1/2}-f_{i-1/2})/dx
    if ( si<TILE ){
      Lu[o] = -(sfu[sj][si+1] - sfu[sj][si])/dx;
    }
  }
}

/*****************/
/* Compute du/dy */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
__global__ void Compute_dG(
  REAL * __restrict__ Lu, 
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny, 
  const REAL dy,
  const int k)
{
  // Shared variables
  __shared__ REAL sfu[TILE+1][WIDTH]; // faces = inner nodes + 1

  // Temporary variables
  REAL u0,u1,u2,u3,u4,u5;
  REAL B0,B1,B2,a0,a1,a2;
  REAL umm,um,uo,up,upp,tau5,asum;

  // Indexes
  unsigned int i,j,si,sj,o; int2 Data;
  
  // local threads indexes
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j =    TILE    * blockIdx.y + threadIdx.y;

  // shared memory indexes
  si = threadIdx.x;
  sj = threadIdx.y;

  // Global index
  o = i+pitch*j+pitch*ny*k;

  if (j < ny){
    // Load data from texture memory
    Data = tex2D(tex_u,i,j-3); u0 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i,j-2); u1 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i,j-1); u2 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i, j ); u3 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i,j+1); u4 = __hiloint2double(Data.y, Data.x);
    Data = tex2D(tex_u,i,j+2); u5 = __hiloint2double(Data.y, Data.x);

    // Reconstruct Face fluxes
    // Split data for f_{i}^{+}
    umm=0.5*(Flux(u0) + fabs(u0)*u0);
    um =0.5*(Flux(u1) + fabs(u1)*u1);
    uo =0.5*(Flux(u2) + fabs(u2)*u2);
    up =0.5*(Flux(u3) + fabs(u3)*u3);
    upp=0.5*(Flux(u4) + fabs(u4)*u4);
    
    // Smooth Indicators (Beta factors)
    B0 = C1312*(umm-2*um+uo )*(umm-2*um +uo ) + 0.25*(umm-4*um+3*uo)*(umm-4*um+3*uo);
    B1 = C1312*(um -2*uo+up )*(um -2*uo +up ) + 0.25*(um-up)*(um-up);
    B2 = C1312*(uo -2*up+upp)*(uo -2*up +upp) + 0.25*(3*uo-4*up+upp)*(3*uo-4*up+upp);
      
    // Alpha weights
    tau5 = fabs(B0-B2);
    a0 = D0N*(1.+tau5/(B0+EPS));
    a1 = D1N*(1.+tau5/(B1+EPS));
    a2 = D2N*(1.+tau5/(B2+EPS));
    asum = a0 + a1 + a2;
    
    // Numerical Flux at cell boundary, $u_{i+1/2}^{-}$;
    sfu[sj][si] = (
      a0*(2*umm- 7*um + 11*uo) +
      a1*( -um + 5*uo + 2*up ) +
      a2*( 2*uo+ 5*up - upp  ))/(6*asum);

    // split data for f_{i}^{-}
    umm=0.5*(Flux(u1) - fabs(u1)*u1);
    um =0.5*(Flux(u2) - fabs(u2)*u2);
    uo =0.5*(Flux(u3) - fabs(u3)*u3);
    up =0.5*(Flux(u4) - fabs(u4)*u4);
    upp=0.5*(Flux(u5) - fabs(u5)*u5);

    // Smooth Indicators (Beta factors)
    B0 = C1312*(umm-2*um+uo )*(umm-2*um +uo ) + 0.25*(umm-4*um+3*uo)*(umm-4*um+3*uo);
    B1 = C1312*(um -2*uo+up )*(um -2*uo +up ) + 0.25*(um-up)*(um-up);
    B2 = C1312*(uo -2*up+upp)*(uo -2*up +upp) + 0.25*(3*uo-4*up+upp)*(3*uo-4*up+upp);
    
    // Alpha weights
    tau5 = fabs(B0-B2);
    a0 = D0P*(1.+tau5/(B0+EPS));
    a1 = D1P*(1.+tau5/(B1+EPS));
    a2 = D2P*(1.+tau5/(B2+EPS));
    asum = a0 + a1 + a2;

    // Numerical Flux at cell boundary, $u_{i+1/2}^{+}$;
    sfu[sj][si] += (
      a0*( -umm + 5*um + 2*uo ) +
      a1*( 2*um + 5*uo - up   ) +
      a2*(11*uo - 7*up + 2*upp))/(6*asum);
    __syncthreads();

    // Compute Lq = (f_{i+1/2}-f_{i-1/2})/dx
    if ( sj<TILE ){
      Lu[o] -= (sfu[sj+1][si] - sfu[sj][si])/dy;
    }
  }
}

/*****************/
/* Compute du/dz */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
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

/********************/
/* Laplace Operator */
/********************/
__global__ void Compute_Laplace(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu,  
  const unsigned int px, // pitch in the x-direction
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz)
{
  REAL above2;
  REAL above;
  REAL center;
  REAL below;
  REAL below2;
  unsigned int i, j, k, o, xy, px2, xy2;
  xy = px*ny; px2 = 2*px; xy2 = 2*xy; 

    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    // For initial slice
    k=3; o=i+px*j+xy*k;

    if (i>2 && i<nx-3 && j>2 && j<ny-3)
    {
      below2=u[o-xy2]; below=u[o-xy]; center=u[o]; above=u[o+xy]; above2=u[o+xy2];

      Lu[o]+= d_kx * (- u[o-2] +16*u[o-1] - 30*center + 16*u[o+1] - u[o+2] ) +
              d_ky * (-u[o-px2]+16*u[o-px]- 30*center + 16*u[o+px]- u[o+px2])+ 
              d_kz * (- below2 +16* below - 30*center + 16* above - above2 );

      // For the rest of the slide
      for(k = 4; k < nz-3; k++)
      {
        o=o+xy; below2=below; below=center; center=above; above=above2; above2=u[o+xy2];

        Lu[o]+= d_kx * (- u[o-2] +16*u[o-1] - 30*center + 16*u[o+1] - u[o+2] ) +
                d_ky * (-u[o-px2]+16*u[o-px]- 30*center + 16*u[o+px]- u[o+px2])+ 
                d_kz * (- below2 +16* below - 30*center + 16* above - above2 );
      }
    }
    // else : do nothing!
}

/**************************/
/* Async Laplace Operator */
/**************************/
__global__ void Compute_Laplace_Async(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu, 
  unsigned int px, // pitch in the x-direction
  unsigned int Nx, 
  unsigned int Ny, 
  unsigned int _Nz, 
  unsigned int kstart, 
  unsigned int kstop, 
  unsigned int loop_z)
{
  register REAL above2;
  register REAL above;
  register REAL center;
  register REAL below;
  register REAL below2;
  unsigned int i, j, k, o, z, XY, px2, XY2;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * loop_z;

  k = MAX(kstart,k);

  XY=px*Ny; px2=px+px; XY2=XY+XY; o=i+px*j+XY*k;

  if (i>2 && i<Nx-3 && j>2 && j<Ny-3)
  {
    below2=u[o-XY2]; below=u[o-XY]; center=u[o]; above=u[o+XY]; above2=u[o+XY2];

    Lu[o]+= d_kx*(- u[o-2] +16* u[o-1]-30*center+16*u[o+1] - u[o+2] ) +
            d_ky*(-u[o-px2]+16*u[o-px]-30*center+16*u[o+px]-u[o+px2]) +
            d_kz*(- below2 +16* below -30*center+16* above - above2 );
      

    for(z = 1; z < loop_z; z++)
    {
      k += 1;

      if (k < MIN(kstop,_Nz+1))
      {
        o=o+XY; below2=below; below=center; center=above; above=above2; above2=u[o+XY2];

        Lu[o]+= d_kx*(- u[o-2] +16*u[o-1] -30*center+16*u[o+1] - u[o+2] ) + 
                d_ky*(-u[o-px2]+16*u[o-px]-30*center+16*u[o+px]-u[o+px2]) + 
                d_kz*(- below2 +16* below -30*center+16* above - above2 );
      }
    }
  }
  // else : do nothing!
}

/***********************/
/* Runge Kutta Methods */  // <==== this is perfectly parallel!
/***********************/
__global__ void Compute_RK( 
  REAL * __restrict__ u, 
  const REAL * __restrict__ uo, 
  const REAL * __restrict__ Lu, 
  const unsigned int step, 
  const unsigned int pitch, 
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz, 
  const REAL dt)
{
  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
    
  // compute single index
  unsigned int o=i+pitch*j+pitch*ny*k;

  // Compute Runge-Kutta step, update only internal cells
  if (i>2 && i<nx-3 && j>2 && j<ny-3 && k>2 && k<nz-3)
  {
    switch (step) {
      case 1: // step 1
          u[o] = uo[o]+dt*(Lu[o]); break;
      case 2: // step 2
          u[o] = 0.75*uo[o]+0.25*(u[o]+dt*(Lu[o])); break;
      case 3: // step 3
          u[o] = (uo[o]+2*(u[o]+dt*(Lu[o])))/3; break;
    }
  }
  // else : do nothing!
}

/*********************/
/* Function Wrappers */
/*********************/
extern "C" void CopyToConstantMemory(const REAL kx, const REAL ky, const REAL kz)
{
  checkCuda(cudaMemcpyToSymbol(d_kx, &kx, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_ky, &ky, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_kz, &kz, sizeof(REAL), 0, cudaMemcpyHostToDevice));
}

extern "C" void InitializeTextures()
{
  // Texture indexing depends on the address Mode
  // index   -7  -6  -5  -4  -3  -2  -1  0  1  2  3  4  5  6  7  8  9  10  11
  // Clamp    0   0   0   0   0   0   0  0  1  2  3  3  3  3  3  3  3   3   3
  // Border   0   0   0   0   0   0   0  0  1  2  3  0  0  0  0  0  0   0   0
  // Wrap     1   2   3   0   1   2   3  0  1  2  3  0  1  2  3  0  1   2   3
  // Mirror   1   2   3   3   2   1   0  0  1  2  3  3  2  1  0  0  1   2   3

  // Texture basic Setup
  tex_u.addressMode[0] = cudaAddressModeClamp;
  tex_u.addressMode[1] = cudaAddressModeClamp;
  tex_u.filterMode = cudaFilterModePoint;
  tex_u.normalized = false;
}

extern "C" void Call_Adv_L(dim3 numBlocks_x, dim3 threadsPerBlock_x, dim3 numBlocks_y, dim3 threadsPerBlock_y, cudaStream_t aStream, 
  size_t pitch_bytes, unsigned int pitch, cudaChannelFormatDesc *desc, unsigned int nx, unsigned int ny, unsigned int nz, REAL dx, REAL dy, REAL *u, REAL *Lu)
{
  for(int k=3; k<=nz-3; k++){
    // Bind texture and Run
    checkCuda(cudaBindTexture2D(0,&tex_u,&u[k*ny*pitch],desc,nx,ny,pitch_bytes));
    Compute_dF<<<numBlocks_x,threadsPerBlock_x,0,aStream>>>(Lu,pitch,nx,ny,dx,k);
    Compute_dG<<<numBlocks_y,threadsPerBlock_y,0,aStream>>>(Lu,pitch,nx,ny,dy,k);
  }
}

// extern "C" void Call_Adv_x(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
//   size_t pitch_bytes, unsigned int pitch, cudaChannelFormatDesc *desc, unsigned int nx, unsigned int ny, unsigned int nz, REAL dx, REAL *u, REAL *Lu)
// {
//   for(int k=3; k<=nz-3; k++){
//     // Bind texture and Run
//     checkCuda(cudaBindTexture2D(0,&tex_u,&u[k*ny*pitch],desc,nx,ny,pitch_bytes));
//     Compute_dF<<<numBlocks,threadsPerBlock,0,aStream>>>(Lu,pitch,nx,ny,dx,k);
//   }
// }

// extern "C" void Call_Adv_y(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
//   size_t pitch_bytes, unsigned int pitch, cudaChannelFormatDesc *desc, unsigned int nx, unsigned int ny, unsigned int nz, REAL dy, REAL *u, REAL *Lu)
// {
//   for(int k=3; k<=nz-3; k++){
//     // Bind Texture and Run
//     checkCuda(cudaBindTexture2D(0,&tex_u,&u[k*ny*pitch],desc,nx,ny,pitch_bytes));
//     Compute_dG<<<numBlocks,threadsPerBlock,0,aStream>>>(Lu,pitch,nx,ny,dy,k);
//   }
// }

extern "C" void Call_Adv_z(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int nx, unsigned int ny, unsigned int nz, REAL dz, REAL *u, REAL *Lu)
{
  Compute_dH<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,nx,ny,nz,dz);
}

extern "C" void Call_Diff_(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream,
  unsigned int pitch, unsigned int nx, unsigned int ny, unsigned int nz, REAL *u, REAL *Lu)
{
  // Compute_Laplace<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,nx,ny,nz);
  Compute_Laplace_Async<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,nx,ny,nz,3,nz-2,LOOP);
}

extern "C" void Call_sspRK(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int step, REAL dt, REAL *u, REAL *uo, REAL *Lu)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(u,uo,Lu,step,pitch,nx,ny,nz,dt);
}
