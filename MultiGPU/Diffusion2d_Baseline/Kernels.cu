//
//  kernels.cu
//  Burgers3d-GPU-MPI
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "DiffusionMPICUDA.h"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

/********************************************/
/* Methods for checking error in CUDA calls */
/********************************************/
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
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
  const unsigned int p) /* p = {0,1} */
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
  const unsigned int p) /* p = {0,1} */
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

/********************/
/* Laplace Operator */
/********************/
__global__ void Compute_Laplace(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu,  
  const REAL kx, 
  const REAL ky, 
  const unsigned int px, // Allocation Pitch
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int jstart,
  const unsigned int jstop)
{
    unsigned int px2=px+px; 

    // global threads
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y + jstart-1;

    // global index
    unsigned int o = i+px*j;

    if (i>2 && i<nx-3 && j<jstop)
    {
      Lu[o] = kx * (-u[ o-2 ]+16*u[o-1 ]-30*u[o]+16*u[o+1 ]-u[ o+2 ]) +
              ky * (-u[o-px2]+16*u[o-px]-30*u[o]+16*u[o+px]-u[o+px2]);
      // Lu[o] = 3; // <-- debuging tool
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
  unsigned int i, j, o;
  // local threads indexes
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;

  // Single index
  o=i+pitch*j; 

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


/*********************/
/* Function Wrappers */
/*********************/
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

extern "C" void Call_sspRK(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int step, unsigned int pitch, unsigned int Nx, unsigned int _Ny, REAL dt, 
  REAL *q, REAL *qo, REAL *Lq)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(q,qo,Lq,step,pitch,Nx,_Ny,dt);
}

extern "C" void Call_Diff_(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  REAL kx, REAL ky, unsigned int pitch, unsigned int Nx, unsigned int _Ny, unsigned int jstart, unsigned int jstop, 
  REAL* q, REAL* Lq)
{
  Compute_Laplace<<<numBlocks,threadsPerBlock,0,aStream>>>(q,Lq,kx,ky,pitch,Nx,_Ny,jstart,jstop);
  CudaCheckError();
}

extern "C" void printGPUmem(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int Nx, unsigned int _Ny,  unsigned int jstart, unsigned int jstop, 
  REAL *q)
{
  PrintGPUmemory<<<numBlocks,threadsPerBlock,0,aStream>>>(q,pitch,Nx,_Ny,jstart,jstop);
}
