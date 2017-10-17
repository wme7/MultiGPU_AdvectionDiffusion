//
//  diffusion3d.h
//  diffusion3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#ifndef acoustics_h
#define acoustics_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* Write solution file */
#define DISPL 0 // Display all error messages
#define WRITE 1 // Write solution to file
#define FLOPS 8.0
#define PI 3.1415926535897932f

/* Define macros */
#define I2D(n,i,j) ((i)+(n)*(j)) // transfrom a 2D array index pair into linear index memory
#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps
#define SINE_DISTRIBUTION(i, j, k, dx, dy, dz) sin(M_PI*i*dx)*sin(M_PI*j*dy)*sin(M_PI*k*dz)
#define EXP_DISTRIBUTION(i, j, k, dx, dy, dz, d, t0) exp( -((-5+i*dx)*(-5+i*dx)+(-5+j*dy)*(-5+j*dy)+(-5+k*dz)*(-5+k*dz))/(4*d*t0) )
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

/* Declare Host Functions */
void Call_Init3D(int IC, REAL *h_u, REAL dx, REAL dy, REAL dz, unsigned int nx, unsigned int ny, unsigned int nz);
void Save3D(REAL *u, unsigned int nx, unsigned int ny, unsigned int nz);
void Print3D(REAL *u, unsigned int nx, unsigned int ny, unsigned int nz);
void SaveBinary3D(REAL *u, unsigned int nx, unsigned int ny, unsigned int nz);

float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz);
void PrintSummary(const char* kernelName, const char* optimization, REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds, int computeIterations, int nx, int ny, int nz);
void CalcError(REAL *u, REAL t, REAL dx, REAL dy, REAL dz, unsigned int nx, unsigned int ny, unsigned int nz);

/* Device wrappers */
extern "C" 
{
void CopyToConstantMemory(REAL kx, REAL ky, REAL kz);
void Call_Lu3d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, unsigned int nx, unsigned int ny, unsigned int nz, REAL *u, REAL *Lu);
void Call_RK3d(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, unsigned int step, unsigned int nx, unsigned int ny, unsigned int nz, REAL dt, REAL *u, REAL *uo, REAL *Lu);
}

#endif /* acoustics_h */
