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

/***********************/
/* WENO RECONSTRUCTION */
/***********************/

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


__device__ REAL WENO5reconstruction(const REAL * __restrict__ u)
{
  REAL B0, B1, B2, a0, a1, a2, alphasum, dflux;
  REAL umm,um,uo,up,upp;

  // Split data for f_{i}^{+}
  umm=C1o2*(Flux(u[0]) + fabs(u[0])*u[0]);
  um =C1o2*(Flux(u[1]) + fabs(u[1])*u[1]);
  uo =C1o2*(Flux(u[2]) + fabs(u[2])*u[2]);
  up =C1o2*(Flux(u[3]) + fabs(u[3])*u[3]);
  upp=C1o2*(Flux(u[4]) + fabs(u[4])*u[4]);
  
  // Smooth Indicators (Beta factors)
  B0 = C13o12*(umm-2*um+uo )*(umm-2*um +uo ) + C1o4*(umm-4*um+3*uo)*(umm-4*um+3*uo);
  B1 = C13o12*(um -2*uo+up )*(um -2*uo +up ) + C1o4*(um-up)*(um-up);
  B2 = C13o12*(uo -2*up+upp)*(uo -2*up +upp) + C1o4*(3*uo-4*up+upp)*(3*uo-4*up+upp);
  
  // Alpha weights
  a0 = D0N/((EPS + B0)*(EPS + B0));
  a1 = D1N/((EPS + B1)*(EPS + B1));
  a2 = D2N/((EPS + B2)*(EPS + B2));
  alphasum = 1./(a0 + a1 + a2);
  
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  dflux =(a0*(2*umm- 7*um + 11*uo) +
          a1*( -um + 5*uo + 2*up) +
          a2*( 2*uo+ 5*up - upp ))*(C1o6*alphasum);

  // split data for f_{i}^{-}
  umm=C1o2*(Flux(u[1]) - fabs(u[1])*u[1]);
  um =C1o2*(Flux(u[2]) - fabs(u[2])*u[2]);
  uo =C1o2*(Flux(u[3]) - fabs(u[3])*u[3]);
  up =C1o2*(Flux(u[4]) - fabs(u[4])*u[4]);
  upp=C1o2*(Flux(u[5]) - fabs(u[5])*u[5]);

  // Smooth Indicators (Beta factors)
  B0 = C13o12*(umm-2*um+uo )*(umm-2*um +uo ) + C1o4*(umm-4*um+3*uo)*(umm-4*um+3*uo);
  B1 = C13o12*(um -2*uo+up )*(um -2*uo +up ) + C1o4*(um-up)*(um-up);
  B2 = C13o12*(uo -2*up+upp)*(uo -2*up +upp) + C1o4*(3*uo-4*up+upp)*(3*uo-4*up+upp);
  
  // Alpha weights
  a0 = D0P/((EPS + B0)*(EPS + B0));
  a1 = D1P/((EPS + B1)*(EPS + B1));
  a2 = D2P/((EPS + B2)*(EPS + B2));
  alphasum = 1./(a0 + a1 + a2);

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  dflux+=(a0*( -umm + 5*um + 2*uo ) +
          a1*( 2*um + 5*uo - up   ) +
          a2*(11*uo - 7*up + 2*upp))*(C1o6*alphasum);
  
  // Compute the numerical flux v_{i+1/2}
  return dflux;
}

__device__ REAL WENO5Zreconstruction(const REAL * __restrict__ u)
{
  REAL B0, B1, B2, a0, a1, a2, tau5, alphasum, dflux;
  REAL umm,um,uo,up,upp;

  // Split data for f_{i}^{+}
  umm=C1o2*(Flux(u[0]) + fabs(u[0])*u[0]);
  um =C1o2*(Flux(u[1]) + fabs(u[1])*u[1]);
  uo =C1o2*(Flux(u[2]) + fabs(u[2])*u[2]);
  up =C1o2*(Flux(u[3]) + fabs(u[3])*u[3]);
  upp=C1o2*(Flux(u[4]) + fabs(u[4])*u[4]);
  
  // Smooth Indicators (Beta factors)
  B0 = C13o12*(umm-2*um+uo )*(umm-2*um +uo ) + C1o4*(umm-4*um+3*uo)*(umm-4*um+3*uo);
  B1 = C13o12*(um -2*uo+up )*(um -2*uo +up ) + C1o4*(um-up)*(um-up);
  B2 = C13o12*(uo -2*up+upp)*(uo -2*up +upp) + C1o4*(3*uo-4*up+upp)*(3*uo-4*up+upp);
  // Alpha weights
  tau5 = fabs(B0-B2);
  a0 = D0N*(1.+tau5/(B0+EPS));
  a1 = D1N*(1.+tau5/(B1+EPS));
  a2 = D2N*(1.+tau5/(B2+EPS));
  alphasum = 1./(a0 + a1 + a2);
  
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  dflux =(a0*(2*umm- 7*um + 11*uo) +
          a1*( -um + 5*uo + 2*up) +
          a2*( 2*uo+ 5*up - upp ))*(C1o6*alphasum);

  // split data for f_{i}^{-}
  umm=C1o2*(Flux(u[1]) - fabs(u[1])*u[1]);
  um =C1o2*(Flux(u[2]) - fabs(u[2])*u[2]);
  uo =C1o2*(Flux(u[3]) - fabs(u[3])*u[3]);
  up =C1o2*(Flux(u[4]) - fabs(u[4])*u[4]);
  upp=C1o2*(Flux(u[5]) - fabs(u[5])*u[5]);

  // Smooth Indicators (Beta factors)
  B0 = C13o12*(umm-2*um+uo )*(umm-2*um +uo ) + C1o4*(umm-4*um+3*uo)*(umm-4*um+3*uo);
  B1 = C13o12*(um -2*uo+up )*(um -2*uo +up ) + C1o4*(um-up)*(um-up);
  B2 = C13o12*(uo -2*up+upp)*(uo -2*up +upp) + C1o4*(3*uo-4*up+upp)*(3*uo-4*up+upp);
  
  // Alpha weights
  tau5 = fabs(B0-B2);
  a0 = D0P*(1.+tau5/(B0+EPS));
  a1 = D1P*(1.+tau5/(B1+EPS));
  a2 = D2P*(1.+tau5/(B2+EPS));
  alphasum = 1./(a0 + a1 + a2);

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  dflux+=(a0*( -umm + 5*um + 2*uo ) +
          a1*( 2*um + 5*uo - up   ) +
          a2*(11*uo - 7*up + 2*upp))*(C1o6*alphasum);
  
  // Compute the numerical flux v_{i+1/2}
  return dflux;
}

/*****************/
/* Compute du/dx */ // <==== parallel strategy: compute serially by rows or by columns!
/*****************/
__global__ void Compute_dF(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu, 
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz, 
  const REAL dx)
{
  // Shared variables
  __shared__ REAL s_u[WIDTH][TILE+6]; // 3-wide halo
  __shared__ REAL sfu[WIDTH][TILE+1]; // faces = inner nodes + 1

  // Indexes
  unsigned int i,j,I,J,K,si,sj,o;
  
  // Global threads indexes
  I =    TILE    * blockIdx.x + threadIdx.x;
  J = blockDim.y * blockIdx.y + threadIdx.y;
  K = blockIdx.z;

  // Local thead indexes
  i = threadIdx.x;
  j = threadIdx.y;

  // Local share memory indexes
  si = threadIdx.x+3; // local i for shared memory access + halo offset
  sj = threadIdx.y;   // local j for shared memory access

  // Global index
  o = I+pitch*J+pitch*ny*K;

  if (I < nx){
    // Load data into shared memory
    s_u[sj][si]=u[o];

    // Load boundary values
    if ( (i<3) && (I<3) ){
      s_u[sj][si-3]=0.; // set Dirichlet BCs
    } else if (i < 3){
      s_u[sj][si-3]=u[o-3]; // get data from neighbour
    }

    // Load boundary values
    if ( (i>TILE-2) && (I>nx-2) ){
      s_u[sj][si+2]=0.; // set Dirichlet BCs
    } else if (i > TILE-2){
      s_u[sj][si+2]=u[o+2]; // get data from neighbour
    }
    __syncthreads();

    // Compute face fluxes
    sfu[j][i]=WENO5Zreconstruction(&s_u[sj][si-3]); // fp_{i+1/2}
    __syncthreads();
      
    // Compute Lq = (f_{i+1/2}-f_{i-1/2})/dx
    if ( i<TILE ){
      Lu[o] = -(sfu[j][i+1] - sfu[j][i])/dx;
    }
  } 
}

/*****************/
/* Compute du/dy */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
__global__ void Compute_dG(
  const REAL * __restrict__ v, 
  REAL * __restrict__ Lv, 
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz, 
  const REAL dy)
{
  // Shared variables
  __shared__ REAL s_v[WIDTH][TILE+6]; // 3-wide halo
  __shared__ REAL sfv[WIDTH][TILE+1]; // faces = inner nodes + 1

  // Indexes
  unsigned int i,j,I,J,K,si,sj,o;
  
  // Global threads indexes
  I = blockDim.x * blockIdx.x + threadIdx.x;
  J =   TILE     * blockIdx.y + threadIdx.y;
  K = blockIdx.z;

  // Local thead indexes
  i = threadIdx.x;
  j = threadIdx.y;

  // Local share memory indexes
  si = threadIdx.x;   // local i for shared memory access
  sj = threadIdx.y+3; // local j for shared memory access + halo offset

  // Global index
  o = I+pitch*J+pitch*ny*K;

  if (J < ny){
    // Load data into shared memory
    s_v[si][sj]=v[o];

    // Load boundary values
    if ( (j<3) && (J<3) ){
      s_v[si][sj-3]=0.; // set Dirichlet BCs
    } else if (j < 3){
      s_v[si][sj-3]=v[o-3*pitch]; // get data from neighbour
    }

    // Load boundary values
    if ( (j>TILE-2) && (J>ny-2) ){
      s_v[si][sj+2]=0.; // set Dirichlet BCs
    } else if (j > TILE-2){
      s_v[si][sj+2]=v[o+2*pitch]; // get data from neighbour
    }
    __syncthreads();

    // Compute face fluxes
    sfv[i][j]=WENO5Zreconstruction(&s_v[si][sj-3]); // fp_{i+1/2}
    __syncthreads();
      
    // Compute Lq = (f_{i+1/2}-f_{i-1/2})/dx
    if ( j<TILE ){
      Lv[o] -= (sfv[i][j+1] - sfv[i][j])/dy;
    }
  }
}

/*****************/
/* Compute du/dz */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
__global__ void Compute_dH(
  const REAL * __restrict__ w, 
  REAL * __restrict__ Lw,
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz, 
  const REAL dz)
{
  // Shared variables
  __shared__ REAL s_w[WIDTH][TILE+6]; // 3-wide halo
  __shared__ REAL sfw[WIDTH][TILE+1]; // faces = inner nodes + 1

  // Indexes
  unsigned int i,k,I,J,K,si,sk,o;

  // Global threads indexes
  I = blockDim.x * blockIdx.x + threadIdx.x;
  K =   TILE     * blockIdx.y + threadIdx.y;
  J = blockIdx.z;

  // Local thead indexes
  i = threadIdx.x;
  k = threadIdx.y;

  // Local share memory indexes
  si = threadIdx.x;   // local i for shared memory access
  sk = threadIdx.y+3; // local j for shared memory access + halo offset

  // Global index
  o = I+pitch*J+pitch*ny*K;

  if (K < nz){
    // Load data into shared memory
    s_w[si][sk]=w[o];

    // Load boundary values
    if ( (k<3) && (K<3) ){
      s_w[si][sk-3]=0.; // set Dirichlet BCs
    } else if (k < 3){
      s_w[si][sk-3]=w[o-3*pitch*ny]; // get data from neighbour
    }

    // Load boundary values
    if ( (k>TILE-2) && (K>nz-2) ){
      s_w[si][sk+2]=0.; // set Dirichlet BCs
    } else if (k > TILE-2){
      s_w[si][sk+2]=w[o+2*pitch*ny]; // get data from neighbour
    }
    __syncthreads();

    // Compute face fluxes
    sfw[i][k]=WENO5Zreconstruction(&s_w[si][sk-3]); // fp_{i+1/2}
    __syncthreads();
      
    // Compute Lq = (f_{i+1/2}-f_{i-1/2})/dz
    if ( k<TILE ){
      Lw[o] -= (sfw[i][k+1] - sfw[i][k])/dz;
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

extern "C" void Call_Adv_x(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int nx, unsigned int ny, unsigned int nz, REAL dx, REAL *u, REAL *Lu)
{
  Compute_dF<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,nx,ny,nz,dx);
}

extern "C" void Call_Adv_y(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int nx, unsigned int ny, unsigned int nz, REAL dy, REAL *u, REAL *Lu)
{
  Compute_dG<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,nx,ny,nz,dy);
}

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
