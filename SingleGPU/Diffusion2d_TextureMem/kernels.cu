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
texture<int2, 2, cudaReadModeElementType> tex_u;

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

__global__ void Compute_Laplace2d_texture(
  REAL * __restrict__ Lu, 
  const unsigned int pitch,
  const unsigned int nx, 
  const unsigned int ny) 
{
  // Threads id
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;  

  int2  o_i2= tex2D(tex_u, i,j ); 
  int2 nn_i2= tex2D(tex_u,i,j+2); 
  int2  n_i2= tex2D(tex_u,i,j+1); 
  int2  s_i2= tex2D(tex_u,i,j-1); 
  int2 ss_i2= tex2D(tex_u,i,j-2); 
  int2 ee_i2= tex2D(tex_u,i+2,j); 
  int2  e_i2= tex2D(tex_u,i+1,j); 
  int2  w_i2= tex2D(tex_u,i-1,j); 
  int2 ww_i2= tex2D(tex_u,i-2,j); 

  double o = __hiloint2double( o_i2.y, o_i2.x); // node( i,j )      nn
  double nn= __hiloint2double(nn_i2.y,nn_i2.x); // node(i,j+2)      |
  double n = __hiloint2double( n_i2.y, n_i2.x); // node(i,j+1)      n
  double s = __hiloint2double( s_i2.y, s_i2.x); // node(i,j-1)      |
  double ss= __hiloint2double(ss_i2.y,ss_i2.x); // node(i,j+1) ee-e-o-w-ww
  double ee= __hiloint2double(ee_i2.y,ee_i2.x); // node(i+2,j)      |
  double e = __hiloint2double( e_i2.y, e_i2.x); // node(i+1,j)      s
  double w = __hiloint2double( w_i2.y, w_i2.x); // node(i-1,j)      |
  double ww= __hiloint2double(ww_i2.y,ww_i2.x); // node(i-2,j)      ss

  // --- Only update "interior" (not boundary) node points
  if (i>2 && i<nx-3 && j>2 && j<ny-3) 
    Lu[i+pitch*j] = d_kx*(-ee+16*e-30*o+16*w-ww) + d_ky*(-nn+16*n-30*o+16*s-ss);
}

/*********************/
/* Function Wrappers */
/*********************/
extern "C" void InitilizeTexture(REAL *d_u, unsigned int nx, unsigned int ny, size_t pitch_bytes)
{
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindSigned);

  cudaBindTexture2D(0, &tex_u, d_u, &desc, nx, ny, pitch_bytes);

  tex_u.addressMode[0] = cudaAddressModeClamp;
  tex_u.addressMode[1] = cudaAddressModeClamp;
  tex_u.filterMode = cudaFilterModePoint;
  tex_u.normalized = false;  
}

extern "C" void CopyToConstantMemory(REAL kx, REAL ky)
{
  checkCuda(cudaMemcpyToSymbol(d_kx, &kx, sizeof(REAL), 0, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpyToSymbol(d_ky, &ky, sizeof(REAL), 0, cudaMemcpyHostToDevice));
}

extern "C" void Call_Lu2d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, unsigned int pitch, unsigned int nx, unsigned int ny, REAL *Lu)
{
  Compute_Laplace2d_texture<<<numBlocks,threadsPerBlock,0,aStream>>>(Lu,pitch,nx,ny);
}

extern "C" void Call_RK2d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, unsigned int step, unsigned int pitch, unsigned int nx, unsigned int ny, REAL dt, REAL *u, REAL *uo, REAL *Lu)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(u,uo,Lu,step,pitch,nx,ny,dt);
}
