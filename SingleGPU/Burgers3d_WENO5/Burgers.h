//
//  Burgers.h
//  Burgers3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#ifndef Burgers_h
#define Burgers_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* WENO constants */
#define D0N 1./10.
#define D1N 6./10.
#define D2N 3./10.
#define D0P 3./10.
#define D1P 6./10.
#define D2P 1./10.
#define EPS 1E-6
#define C1312 13./12.
#define C14 1./4.

/* Write solution file */
#define DISPL 0 // Display all error messages
#define WRITE 0 // Write solution to file
#define FLOPS 8.0

/* Length for Blocking strategy */
#define LOOP 16

/* Define macros */
#define I2D(n,i,j) ((i)+(n)*(j)) // transfrom a 2D array index pair into linear index memory
#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps
#define GAUSSIAN_DISTRIBUTION(x,y,z) 1.0*exp(-((x-1.0)*(x-1.0)+(y-1.0)*(y-1.0)+(z-1.0)*(z-1.0))/0.1)
#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* Use floats of doubles */
#define USE_FLOAT false // set false to use double
#if USE_FLOAT
	#define REAL	float
	#define MPI_CUSTOM_REAL MPI_REAL
#else
	#define REAL	double
	#define MPI_CUSTOM_REAL MPI_DOUBLE
#endif

/******************/
/* Host functions */
/******************/
void Call_Init3d(int IC, REAL *u, REAL dx, REAL dy, REAL dz, unsigned int Nx, unsigned int Ny, unsigned int Nz);
void Save3D(REAL *u, unsigned int Nx, unsigned int Ny, unsigned int Nz);
void Print3D(REAL *u, unsigned int Nx, unsigned int Ny, unsigned int Nz);
void SaveBinary3D(REAL *u, unsigned int Nx, unsigned int Ny, unsigned int Nz);

float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz);
void PrintSummary(const char* kernelName, const char* optimization, double computeTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, float gflops, const int computeIterations, unsigned int Nx, unsigned int Ny, unsigned int Nz);

/*******************/
/* Device wrappers */
/*******************/
extern "C" 
{
void CopyToConstantMemory(REAL kx, REAL ky, REAL kz);
void Call_Adv_x(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
	unsigned int Nx, unsigned int Ny, unsigned int Nz, REAL dx, REAL *u, REAL *Lu);
void Call_Adv_y(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
	unsigned int Nx, unsigned int Ny, unsigned int Nz, REAL dy, REAL *v, REAL *Lv);
void Call_Adv_z(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
	unsigned int Nx, unsigned int Ny, unsigned int Nz, REAL dz, REAL *w, REAL *Lw);
void Call_Diff_(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream,
	unsigned int Nx, unsigned int Ny, unsigned int Nz, REAL *q, REAL *Lq);
void Call_sspRK(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
	unsigned int Nx, unsigned int Ny, unsigned int Nz, unsigned int step, REAL dt, REAL *u, REAL *uo, REAL *Lu);
}

#endif /* Burgers_h */
