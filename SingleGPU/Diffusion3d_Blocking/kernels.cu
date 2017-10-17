//
//  kernels.cu
//  diffusion3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

extern "C" {
#include "acoustics3d.h"
}

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

__constant__ REAL d_kx;
__constant__ REAL d_ky;
__constant__ REAL d_kz;

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

/********************/
/* Laplace Operator */
/********************/
__global__ void Compute_Laplace3d_Async(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu, 
  const unsigned int px, // allocation pitch
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int _Nz, 
  const unsigned int kstart, 
  const unsigned int kstop, 
  const unsigned int loop_z)
{
  register REAL above2;
  register REAL above;
  register REAL center;
  register REAL below;
  register REAL below2;
  unsigned int z, XY, Nx2, XY2;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * loop_z;

  k = MAX(kstart,k);

  XY=px*Ny; Nx2=px+px; XY2=XY+XY; 

  int o=i+px*j+XY*k;

  if (i>2 && i<Nx-3 && j>2 && j<Ny-3)
  {
    below2=u[o-XY2]; below=u[o-XY]; center=u[o]; above=u[o+XY]; above2=u[o+XY2];

    Lu[o] = d_kx * (- u[o-2] + 16*u[o-1] - 30*center + 16*u[o+1] - u[o+2]) + 
            d_ky * (-u[o-Nx2]+ 16*u[o-px]- 30*center + 16*u[o+px]- u[o+Nx2]) +
            d_kz * (- below2 + 16*below  - 30*center + 16* above - above2 );

    for(z = 1; z < loop_z; z++)
    {
      k += 1;

      if (k < MIN(kstop,_Nz+1))
      {
        o=o+XY; below2=below; below=center; center=above; above=above2; above2=u[o+XY2];

        Lu[o] = d_kx * (- u[o-2] + 16*u[o-1] - 30*center + 16*u[o+1] - u[o+2]) +
                d_ky * (-u[o-Nx2]+ 16*u[o-px]- 30*center + 16*u[o+px]- u[o+Nx2]) +
                d_kz * (- below2 + 16* below - 30*center + 16* above - above2 );
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
  // Compute Runge-Kutta step, local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
    
  // compute single index
  int o=i+pitch*j+pitch*ny*k;

  // update only internal cells
  if (i>1 && i<nx-2 && j>1 && j<ny-2 && k>1 && k<nz-2)
  {
    switch (step) {
      case 1: // step 1
          u[o] = uo[o]+dt*(Lu[o]);
          break;
      case 2: // step 2
          u[o] = 0.75*uo[o]+0.25*(u[o]+dt*(Lu[o]));
          break;
      case 3: // step 3
          u[o] = (uo[o]+2*(u[o]+dt*(Lu[o])))/3;
          break;
    }
  }
  // else : do nothing!
}

__global__ void Compute_RK_Async( 
  REAL * __restrict__ q, 
  const REAL * __restrict__ qo, 
  const REAL * __restrict__ Lq, 
  const unsigned int step,
  const unsigned int pitch, 
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int _Nz, 
  const unsigned int kstart, 
  const unsigned int kstop, 
  const unsigned int loop_z, 
  const REAL dt)
{
  int z, XY = pitch*Ny;
  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockIdx.z * loop_z;

  k = MAX(kstart,k);

  // Single index
  int o=i+pitch*j+XY*k;

  // Compute Runge-Kutta step only on internal cells
  if (i>1 && i<Nx-2 && j>1 && j<Ny-2)
  {
    for(z = 0; z < loop_z; z++)
    {
      if (k < MIN(kstop,_Nz-2)) 
      {
        switch (step) {
          case 1: // step 1
            q[o] = qo[o]+dt*(Lq[o]); break;
          case 2: // step 2
            q[o] = 0.75*qo[o]+0.25*(q[o]+dt*(Lq[o])); break;
          case 3: // step 3
            q[o] = (qo[o]+2*(q[o]+dt*(Lq[o])))/3; break;
        }
        o += XY;
      }
      k += 1;
    }
  }
}

/*********************/
/* Function Wrappers */
/*********************/
extern "C" void CopyToConstantMemory(REAL kx, REAL ky, REAL kz)
{
  checkCuda(cudaMemcpyToSymbol(d_kx, &kx, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_ky, &ky, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_kz, &kz, sizeof(REAL), 0, cudaMemcpyHostToDevice));
}

extern "C" void Call_Lu3d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int nx, unsigned int ny, unsigned int nz, REAL *q, REAL *Lq)
{
  Compute_Laplace3d_Async<<<numBlocks,threadsPerBlock,0,aStream>>>(q,Lq,pitch,nx,ny,nz,3,nz-2,k_loop);
}

extern "C" void Call_RK3d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int step, unsigned int pitch, unsigned int nx, unsigned int ny, unsigned int nz, REAL dt, REAL *q, REAL *qo, REAL *Lq)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(q,qo,Lq,step,pitch,nx,ny,nz,dt);
  // Compute_RK_Async<<<numBlocks,threadsPerBlock,0,aStream>>>(q,qo,Lq,step,pitch,nx,ny,nz,3,nz-2,k_loop,dt);
}
