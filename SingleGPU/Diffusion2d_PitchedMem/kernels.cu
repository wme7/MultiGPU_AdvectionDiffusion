//
//  kernels.cu
//  diffusion2d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

extern "C" {
#include "diffusion2d.h"
}

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

__constant__ REAL d_kx;
__constant__ REAL d_ky;

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

/***********************/
/* Runge Kutta Methods */  
/***********************/
__global__ void Compute_RK( 
  REAL * __restrict__ u, 
  const REAL * __restrict__ uo, 
  const REAL * __restrict__ Lu, 
  const unsigned int step, 
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny, 
  const REAL dt){

  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
    
  // Compute Runge-Kutta step
  // compute single index
  unsigned int o=i+pitch*j;
  // update only internal cells
  if (i>2 && i<nx-3 && j>2 && j<ny-3)
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
  // else do nothing!
}

__global__ void Compute_Laplace2d(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu, 
  const unsigned int px, // allocation picth
  const unsigned int nx, 
  const unsigned int ny)
{
  unsigned int i, j, o;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    o = i+px*j;

    if (i>2 && i<nx-3 && j>2 && j<ny-3)
      Lu[o] = d_kx*(-  u[o-2]  +16*u[o-1] -30*u[o]+16*u[o+1] -  u[o+2]  )+
              d_ky*(-u[o-px-px]+16*u[o-px]-30*u[o]+16*u[o+px]-u[o+px+px]);
    else
      Lu[o] = 0.0;
}

/*********************/
/* Function Wrappers */
/*********************/
extern "C" void CopyToConstantMemory(const REAL kx, const REAL ky)
{
  checkCuda(cudaMemcpyToSymbol(d_kx, &kx, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_ky, &ky, sizeof(REAL), 0, cudaMemcpyHostToDevice));
}

extern "C" void Call_Lu2d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int nx, unsigned int ny, REAL *u, REAL *Lu)
{
  Compute_Laplace2d<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,pitch,nx,ny);
}

extern "C" void Call_RK2d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int step, unsigned int pitch, unsigned int nx, unsigned int ny, REAL dt, REAL *u, REAL *uo, REAL *Lu)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(u,uo,Lu,step,pitch,nx,ny,dt);
}
