//
//  DiffusionMPICUDA.h
//  Diffusion2d-GPU-MPI
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#ifndef _DIFFUSION_GPU_MPI_H__
#define _DIFFUSION_GPU_MPI_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* WENO constants */
#define D0N 1.0/10.0
#define D1N 6.0/10.0
#define D2N 3.0/10.0
#define D0P 3.0/10.0
#define D1P 6.0/10.0
#define D2P 1.0/10.0
#define EPS 1E-6
#define C1312 13.0/12.0
#define C14 1.0/4.0

// Testing :
// A grid of n subgrids
  /* bottom
  +-------+ 
  | 0 (0) | mpi_rank (gpu)
  +-------+
  | 1 (1) |
  +-------+
     ...
  +-------+
  | n (n) |
  +-------+
    top */

/*************/
/* Constants */
/*************/
#define DEBUG 0 // Display all error messages
#define WRITE 1 // Write solution to file
#define RADIUS 3 // gosh cells
#define LOOP 8 // for asyncronous kernel implementation
#define FLOPS 8.0 // Double Precision
#define ROOT 0 // Define root process

/* Define macros */
#define I2D(n,i,j) ((i)+(n)*(j)) // transfrom a 2D array index pair into linear index memory
#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps
#define GAUSSIAN_DISTRIBUTION(x,y) 1.0*exp(-((x*x)+(y*y))/0.1)
#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { printf("MPI error calling \""#call"\"\n"); exit(-1); }

/* use floats of dobles */
#define USE_FLOAT false // set false to use real
#if USE_FLOAT
	#define REAL	float
	#define MPI_CUSTOM_REAL MPI_FLOAT
#else
	#define REAL	double
	#define MPI_CUSTOM_REAL MPI_DOUBLE
#endif

/* enviroment variable */
#define USE_OMPI true // set false for MVAPICH2
#if USE_OPMI
	#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"
#else
	#define ENV_LOCAL_RANK "MV2_COMM_WORLD_LOCAL_RANK"
#endif

/******************/
/* Host functions */
/******************/
void Print2D(REAL *u, unsigned int nx, unsigned int ny);
void Save_2D(REAL *u, unsigned int nx, unsigned int ny);
void SaveBinary2D(REAL *u, unsigned int nx, unsigned int ny, const char *name);
void Init_domain(int IC, REAL *u0, REAL dx, REAL dy, unsigned int nx, unsigned int ny);
void Init_subdomain(REAL *h_q, REAL *h_s_q, unsigned int n, unsigned int Nx, unsigned int _Ny);
void Merge_domains(REAL *h_s_q, REAL *h_q, unsigned int n, unsigned int Nx, unsigned int _Ny);
void InitializeMPI(int* argc, char*** argv, int* rank, int* numberOfProcesses);
void FinalizeMPI();
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny);
void PrintSummary(const char* kernelName, const char* optimization, double computeTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, float gflops, int computeIterations, unsigned int nx, unsigned int ny);

/*******************/
/* Device wrappers */
/*******************/
extern "C"
{
	int DeviceScan();
	int getBlock(int n, int block);
	void ECCCheck(int rank);
	void AssignDevices(int rank);

	void CopyBoundaryRegionToGhostCellAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
		REAL* d_s_q, REAL* d_send_buffer,
		unsigned int Nx, unsigned int _Ny, unsigned int pitch, unsigned int gc_pitch, unsigned int side);
	void CopyGhostCellToBoundaryRegionAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
		REAL* d_s_q, REAL* d_recv_buffer, 
		unsigned int Nx, unsigned int _Ny, unsigned int pitch, unsigned int gc_pitch, unsigned int side);
	void Call_Diff_(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream,
		REAL kx, REAL ky, unsigned int pitch, unsigned int Nx, unsigned int _Ny, unsigned int kstart, unsigned int kstop, 
		REAL* d_s_q, REAL* d_s_Lq);
	void Call_sspRK(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
		unsigned int step, unsigned int pitch, unsigned int Nx, unsigned int _Ny, REAL dt, 
		REAL *q, REAL *qo, REAL *Lq);
	void printGPUmem(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
		unsigned int pitch, unsigned int Nx, unsigned int _Ny, unsigned int kstart, unsigned int kstop, 
		REAL *d_s_q);
}

#endif	// _DIFFUSION_GPU_MPI_H__
