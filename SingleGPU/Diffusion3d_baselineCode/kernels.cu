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
__global__ void Compute_Laplace3d_O2(
  REAL * __restrict__ u, 
  REAL * __restrict__ Lu,  
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz)
{
  unsigned int i, j, k, o, n, s, e, w, t, b, xy;
  xy = nx*ny;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;
    k = threadIdx.z + blockIdx.z * blockDim.z;

    o = i+(nx*j)+(xy*k); // node( j,i,k )      n  b
    n = o+nx;            // node(j+1,i,k)      | /
    s = o-nx;            // node(j-1,i,k)      |/ 
    e = o+1;             // node(j,i+1,k)  w---o---e
    w = o-1;             // node(j,i-1,k)     /|
    t = o+xy;            // node(j,i,k+1)    / |
    b = o-xy;            // node(j,i,k-1)   t  s
    
    if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1)
      Lu[o] = 12*d_kx*(u[e]-2*u[o]+u[w])+
              12*d_ky*(u[n]-2*u[o]+u[s])+
              12*d_kz*(u[t]-2*u[o]+u[b]);
    // else : do nothing!
}

__global__ void Compute_Laplace3d(
  REAL * __restrict__ u, 
  REAL * __restrict__ Lu,  
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz)
{
  unsigned int i, j, k, o, n, s, e, w, t, b, nn, ss, ee, ww, tt, bb, xy;
  xy = nx*ny;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;
    k = threadIdx.z + blockIdx.z * blockDim.z;

    o = i+(nx*j)+(xy*k); // node( j,i,k )        nn    bb
    nn= o+nx+nx;         // node(j+2,i,k)        |    /
    n = o+nx;            // node(j+1,i,k)        |   /
    s = o-nx;            // node(j-1,i,k)        n  b
    ss= o-nx-nx;         // node(j-2,i,k)        | /
    ee= o+2;             // node(j,i+2,k)        |/
    e = o+1;             // node(j,i+1,k) ww--w--o--e--ee
    w = o-1;             // node(j,i-1,k)       /|
    ww= o-2;             // node(j,i-2,k)      / |
    tt= o+xy+xy;         // node(j,i,k+2)     t  s
    t = o+xy;            // node(j,i,k+1)    /   |
    b = o-xy;            // node(j,i,k-1)   /    |
    bb= o-xy-xy;         // node(j,i,k-2)  tt    ss

    if (i>1 && i<nx-2 && j>1 && j<ny-2 && k>1 && k<nz-2)
      Lu[o] = d_kx*(-u[ee]+16*u[e]-30*u[o]+16*u[w]-u[ww])+ 
              d_ky*(-u[nn]+16*u[n]-30*u[o]+16*u[s]-u[ss])+ 
              d_kz*(-u[tt]+16*u[t]-30*u[o]+16*u[b]-u[bb]);
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
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz, 
  const REAL dt)
{
  // Using (o) = [i+nx*j+nx*ny*k] index
  unsigned int i, j, k, o, xy; 
  xy=nx*ny;

  // Compute Runge-Kutta step, local threads indexes
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  k = blockDim.z * blockIdx.z + threadIdx.z;
    
  // compute single index
  o=i+nx*j+xy*k;

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

/*********************/
/* Function Wrappers */
/*********************/
extern "C" void CopyToConstantMemory(const REAL kx, const REAL ky, const REAL kz)
{
  checkCuda(cudaMemcpyToSymbol(d_kx, &kx, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_ky, &ky, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_kz, &kz, sizeof(REAL), 0, cudaMemcpyHostToDevice));
}

extern "C" void Call_Lu3d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int nx, unsigned int ny, unsigned int nz, REAL *u, REAL *Lu)
{
  Compute_Laplace3d<<<numBlocks,threadsPerBlock,0,aStream>>>(u,Lu,nx,ny,nz);
}

extern "C" void Call_RK3d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int step, unsigned int nx, unsigned int ny, unsigned int nz, const REAL dt, REAL *u, REAL *uo, REAL *Lu)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(u,uo,Lu,step,nx,ny,nz,dt);
}
