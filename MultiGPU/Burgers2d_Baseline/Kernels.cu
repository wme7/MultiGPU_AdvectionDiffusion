//
//  kernels.cu
//  Burgers3d-GPU-MPI
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "BurgersMPICUDA.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

/*********************************************/
/* A method for checking error in CUDA calls */
/*********************************************/
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

/******************************/
/* Kernel for computing halos */
/******************************/
__global__ void copy_br_to_gc(
  const REAL * __restrict__ un, 
  REAL * __restrict__ gc_un, 
  const unsigned int NX, 
  const unsigned int _NY, 
  const unsigned int pitch, 
  const unsigned int gc_pitch, 
  const unsigned int p /* p = {0,1} */) 
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned int j0 = _NY-6 + p*(3-_NY+6); // {0,1}: j1 = {3,(Ny-1)-5}
  unsigned int j1 = _NY-5 + p*(4-_NY+5); // {0,1}: j2 = {4,(Ny-1)-4}
  unsigned int j2 = _NY-4 + p*(5-_NY+4); // {0,1}: j3 = {5,(Ny-1)-3}
  unsigned int ibr0 = i + j0*pitch;
  unsigned int ibr1 = i + j1*pitch;
  unsigned int ibr2 = i + j2*pitch;
  unsigned int igc0 = i;//0*gc_xy;
  unsigned int igc1 = i + 1*gc_pitch;
  unsigned int igc2 = i + 2*gc_pitch;

  if( i < NX && j2 < _NY)
  {
    gc_un[igc0] = un[ibr0];
    gc_un[igc1] = un[ibr1];
    gc_un[igc2] = un[ibr2];
  }
}

/******************************/
/* Kernel for computing halos */
/******************************/
__global__ void copy_gc_to_br(
  REAL * __restrict__ un, 
  const REAL * __restrict__ gc_un, 
  const unsigned int NX, 
  const unsigned int _NY, 
  const unsigned int pitch, 
  const unsigned int gc_pitch, 
  const unsigned int p /* p = {0,1} */)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned int j0 = _NY-3 + p*( -_NY+3); // {0,1}: j1 = {0,(Ny-1)-2}
  unsigned int j1 = _NY-2 + p*(1-_NY+2); // {0,1}: j2 = {1,(Ny-1)-1}
  unsigned int j2 = _NY-1 + p*(2-_NY+1); // {0,1}: j3 = {2,(Ny-1)-0}
  unsigned int igc0 = i;//0*gc_xy;
  unsigned int igc1 = i + 1*gc_pitch;
  unsigned int igc2 = i + 2*gc_pitch;
  unsigned int ibr0 = i + j0*pitch;
  unsigned int ibr1 = i + j1*pitch;
  unsigned int ibr2 = i + j2*pitch;

  if( i < NX && j2 < _NY )
  {
    un[ibr0] = gc_un[igc0];
    un[ibr1] = gc_un[igc1];
    un[ibr2] = gc_un[igc2];
  }
}

/*****************/
/* FLUX FUNCTION */
/*****************/
__device__ REAL Flux(const REAL u){
  return 0.5*u*u;
}

/***********************/
/* WENO RECONSTRUCTION */
/***********************/
__device__ REAL Reconstruct1d(
  const REAL vmm,
  const REAL vm,
  const REAL v,
  const REAL vp,
  const REAL vpp,
  const REAL umm,
  const REAL um,
  const REAL u,
  const REAL up,
  const REAL upp){
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
  REAL B0n, B1n, B2n, B0p, B1p, B2p;
  REAL w0n, w1n, w2n, w0p, w1p, w2p;
  REAL a0n, a1n, a2n, a0p, a1p, a2p;
  REAL alphasumn, alphasump, hn, hp;
  
  // Smooth Indicators (Beta factors)
  B0n = C13o12*(vmm-2*vm+v  )*(vmm-2*vm+v  ) + C1o4*(vmm-4*vm+3*v)*(vmm-4*vm+3*v);
  B1n = C13o12*(vm -2*v +vp )*(vm -2*v +vp ) + C1o4*(vm-vp)*(vm-vp);
  B2n = C13o12*(v  -2*vp+vpp)*(v  -2*vp+vpp) + C1o4*(3*v-4*vp+vpp)*(3*v-4*vp+vpp);
  
  // Alpha weights
  a0n = D0N/((EPS + B0n)*(EPS + B0n));
  a1n = D1N/((EPS + B1n)*(EPS + B1n));
  a2n = D2N/((EPS + B2n)*(EPS + B2n));
  alphasumn = a0n + a1n + a2n;
  
  // ENO stencils weigths
  w0n = a0n/alphasumn;
  w1n = a1n/alphasumn;
  w2n = a2n/alphasumn;
  
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  hn = (w0n*(2*vmm- 7*vm + 11*v) +
        w1n*( -vm + 5*v  + 2*vp) +
        w2n*( 2*v + 5*vp - vpp ))/6;

  // Smooth Indicators (Beta factors)
  B0p = C13o12*(umm-2*um+u  )*(umm-2*um +u  ) + C1o4*(umm-4*um+3*u)*(umm-4*um+3*u);
  B1p = C13o12*(um -2*u +up )*(um -2*u  +up ) + C1o4*(um-up)*(um-up);
  B2p = C13o12*(u  -2*up+upp)*(u  -2*up +upp) + C1o4*(3*u-4*up+upp)*(3*u-4*up+upp);
  
  // Alpha weights
  a0p = D0P/((EPS + B0p)*(EPS + B0p));
  a1p = D1P/((EPS + B1p)*(EPS + B1p));
  a2p = D2P/((EPS + B2p)*(EPS + B2p));
  alphasump = a0p + a1p + a2p;
  
  // ENO stencils weigths
  w0p = a0p/alphasump;
  w1p = a1p/alphasump;
  w2p = a2p/alphasump;

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  hp = (w0p*( -umm + 5*um + 2*u  ) +
        w1p*( 2*um + 5*u  - up   ) +
        w2p*(11*u  - 7*up + 2*upp))/6;
  
  // Compute the numerical flux v_{i+1/2}
  return (hn+hp);
}

/*****************/
/* Compute dF/dx */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
__global__ void Compute_dF(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu, 
  const unsigned int pitch, // allocation pitch
  const unsigned int nx, 
  const unsigned int _ny, 
  const unsigned int jstart, 
  const unsigned int jstop, 
  const REAL dx)
{
  // Temporary variables
  REAL fu, fu_old;
  REAL f1mm, f1m, f1, f1p, f1pp;
  REAL g1mm, g1m, g1, g1p, g1pp;

  // Indexes
  unsigned int i, j, o;
  
  // local threads indexes
  j = blockDim.y * blockIdx.y + threadIdx.y;

  // Compute only for internal nodes
  if (j>=jstart && j<jstop) 
  {
    o=pitch*j;
    // Old resulst arrays
    fu_old=0;
    
    f1mm= 0.5*(Flux(u[ o ]) + fabs(u[ o ])*u[ o ]); // node(i-2)
    f1m = 0.5*(Flux(u[1+o]) + fabs(u[1+o])*u[1+o]); // node(i-1)
    f1  = 0.5*(Flux(u[2+o]) + fabs(u[2+o])*u[2+o]); // node( i )     imm--im--i--ip--ipp--ippp
    f1p = 0.5*(Flux(u[3+o]) + fabs(u[3+o])*u[3+o]); // node(i+1)
       
    g1mm= 0.5*(Flux(u[1+o]) - fabs(u[1+o])*u[1+o]); // node(i-1)
    g1m = 0.5*(Flux(u[2+o]) - fabs(u[2+o])*u[2+o]); // node( i )     imm--im--i--ip--ipp--ippp
    g1  = 0.5*(Flux(u[3+o]) - fabs(u[3+o])*u[3+o]); // node(i+1)
    g1p = 0.5*(Flux(u[4+o]) - fabs(u[4+o])*u[4+o]); // node(i+2)
    
    for (i = 2; i < nx-3; i++) 
    {
      // Compute and split fluxes
      f1pp= 0.5*(Flux(u[i+2+o]) + fabs(u[i+2+o])*u[i+2+o]); // node(i+2)
      g1pp= 0.5*(Flux(u[i+3+o]) - fabs(u[i+3+o])*u[i+3+o]); // node(i+3)
      
      // Reconstruct
      fu = Reconstruct1d(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);
      
      // Compute Lq = dF/dx
      Lu[i+o]=-(fu-fu_old)/dx; // dudx
      //Lu[i+o]=i+o; // <-- debuging tool
      
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

/*****************/
/* Compute dG/dy */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
__global__ void Compute_dG(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu, 
  const unsigned int pitch, // allocation pnxch
  const unsigned int nx, 
  const unsigned int _ny, 
  const unsigned int jstart, 
  const unsigned int jstop, 
  const REAL dy)
{
  ///Temporary variables
  REAL fu, fu_old;
  REAL f1mm, f1m, f1, f1p, f1pp;
  REAL g1mm, g1m, g1, g1p, g1pp;

  // Indexes
  unsigned int i, j, o, nj; nj = jstop-jstart;
  
  // nxcal threads innxxes
  i = blockDim.x * blockIdx.x + threadIdx.x;

  // Compute only for internal nodes
  if (i>=3 && i<nx-3) 
  {
    o=i+pitch*jstart;
    // Old resulst arrays
    f1mm= 0.5*(Flux(u[o-3*pitch]) + fabs(u[o-3*pitch])*u[o-3*pitch]); // node(i-2)
    f1m = 0.5*(Flux(u[o-2*pitch]) + fabs(u[o-2*pitch])*u[o-2*pitch]); // node(i-1)
    f1  = 0.5*(Flux(u[ o-pitch ]) + fabs(u[ o-pitch ])*u[ o-pitch ]); // node( i )     imm--im--i--ip--ipp--ippp
    f1p = 0.5*(Flux(u[   o     ]) + fabs(u[   o     ])*u[   o     ]); // node(i+1)
    f1pp= 0.5*(Flux(u[ o+pitch ]) + fabs(u[ o+pitch ])*u[ o+pitch ]); // node(i+1)
    
    g1mm= 0.5*(Flux(u[o-2*pitch]) - fabs(u[o-2*pitch])*u[o-2*pitch]); // node(i-1)
    g1m = 0.5*(Flux(u[ o-pitch ]) - fabs(u[ o-pitch ])*u[ o-pitch ]); // node( i )     imm--im--i--ip--ipp--ippp
    g1  = 0.5*(Flux(u[  o      ]) - fabs(u[  o      ])*u[  o      ]); // node(i+1)
    g1p = 0.5*(Flux(u[ o+pitch ]) - fabs(u[ o+pitch ])*u[ o+pitch ]); // node(i+2)
    g1pp= 0.5*(Flux(u[o+2*pitch]) - fabs(u[o+2*pitch])*u[o+2*pitch]); // node(i+2)

    fu_old = Reconstruct1d(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);

    f1mm=f1m; f1m=f1; f1=f1p; f1p=f1pp;
    g1mm=g1m; g1m=g1; g1=g1p; g1p=g1pp;
        
    for (j = 0; j < nj; j++) 
    {
      // Compute and split fluxes
      f1pp= 0.5*(Flux(u[o+(j+2)*pitch]) + fabs(u[o+(j+2)*pitch])*u[o+(j+2)*pitch]); // node(i+2)
      g1pp= 0.5*(Flux(u[o+(j+3)*pitch]) - fabs(u[o+(j+3)*pitch])*u[o+(j+3)*pitch]); // node(i+3)
      
      // Reconstruct
      fu = Reconstruct1d(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);
      
      // Compute Lq = dG/dy
      Lu[o+pitch*j]-=(fu-fu_old)/dy; // dudy
      //Lu[o+pitch*j]=o+[pitch]*j; // <-- debuging tool
      
      // Save old results
      fu_old=fu;

      f1mm=f1m; f1m=f1; f1=f1p; f1p=f1pp;
      g1mm=g1m; g1m=g1; g1=g1p; g1p=g1pp;
    }
  }
}

/***********************/
/* Runge Kutta Methods */  // <==== this is perfectly parallel!
/***********************/
__global__ void Compute_RK( 
  REAL * __restrict__ q, 
  const REAL * __restrict__ qo, 
  const REAL * __restrict__ Lq, 
  const unsigned int step,
  const unsigned int pitch, 
  const unsigned int Nx,
  const unsigned int Ny,
  const REAL dt)
{
  // Global threads indexes
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

  // Global index
  unsigned int o=i+pitch*j; 

  // Compute Runge-Kutta step only on internal cells
  if (i < Nx && j < Ny)
  {
    switch (step) {
      case 1: // step 1
        q[o] = qo[o]+dt*Lq[o]; break;
      case 2: // step 2
        q[o] = 0.75*qo[o]+0.25*(q[o]+dt*Lq[o]); break;
      case 3: // step 3
        q[o] = (qo[o]+2*(q[o]+dt*Lq[o]))/3; break;
    }
    // q[o] = o; // <-- debuging tool
  }
}

/********************/
/* Print GPU memory */ 
/********************/
__global__ void PrintGPUmemory( 
  REAL * __restrict__ q, 
  const unsigned int pitch, 
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int jstart, 
  const unsigned int jstop)
{
  unsigned int i, j, o;

  printf("jstart: %d,\t jstop: %d\n",jstart,jstop);

  // Print only on internal cells
  for (j = 0; j < Ny; j++) 
  {
    for (i = 0; i < Nx; i++) 
    {
      o=i+pitch*j; printf("%8.2f", q[o]); 
    }
    printf("\n"); 
  }
}


/******************************************************************************/
/* Function that copies content from the host to the device's constant memory */
/******************************************************************************/
extern "C" void CopyBoundaryRegionToGhostCellAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
  REAL* d_s_q, REAL* d_send_buffer, unsigned int Nx, unsigned int _Ny, unsigned int pitch, unsigned int gc_pitch, unsigned int p)
{
  copy_br_to_gc<<<thread_blocks_halo,threads_per_block,0,aStream>>>(d_s_q,d_send_buffer,Nx,_Ny,pitch,gc_pitch,p);
}

extern "C" void CopyGhostCellToBoundaryRegionAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
  REAL* d_s_q, REAL* d_recv_buffer, unsigned int Nx, unsigned int _Ny, unsigned int pitch, unsigned int gc_pitch, unsigned int p)
{
  copy_gc_to_br<<<thread_blocks_halo,threads_per_block,0,aStream>>>(d_s_q,d_recv_buffer,Nx,_Ny,pitch,gc_pitch,p);
}

extern "C" void Compute_Adv_x_Async(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int Nx, unsigned int _Ny, unsigned int jstart, unsigned int jstop, REAL dx, 
  REAL *u, REAL *Lu)
{
  Compute_dF<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,Nx,_Ny,jstart,jstop,dx);
}

extern "C" void Compute_Adv_y_Async(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int Nx, unsigned int _Ny, unsigned int jstart, unsigned int jstop, REAL dy,
  REAL *u, REAL *Lu)
{
  Compute_dG<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,Nx,_Ny,jstart,jstop,dy);
}

extern "C" void Compute_sspRK3Async(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int step, unsigned int pitch, unsigned int Nx, unsigned int _Ny, REAL dt, 
  REAL *u, REAL *uo, REAL *Lu)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(u,uo,Lu,step,pitch,Nx,_Ny,dt);
}

extern "C" void printGPUmem(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int Nx, unsigned int _Ny,  unsigned int jstart, unsigned int jstop, 
  REAL *q)
{
  PrintGPUmemory<<<numBlocks,threadsPerBlock,0,aStream>>>(q,pitch,Nx,_Ny,jstart,jstop);
}
