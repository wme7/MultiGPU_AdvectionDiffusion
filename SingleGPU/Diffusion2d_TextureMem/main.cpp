//
//  main.cpp
//  diffusion2d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "diffusion2d.h"
#include <time.h>

//  Define a method for checking error in CUDA calls
#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)
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

//  Main Program
int main(int argc, char** argv)
{
    REAL C;
    unsigned int L, W, Nx, Ny, max_iters, blockX, blockY;

    if (argc == 9)
    {
        C = atof(argv[1]); // conductivity, here it is assumed: Cx = Cy = C.
        L = atoi(argv[2]); // domain lenght
        W = atoi(argv[3]); // domain width 
        Nx = atoi(argv[4]); // number cells in x-direction
        Ny = atoi(argv[5]); // number cells in x-direction
        max_iters = atoi(argv[6]); // number of iterations / time steps
        blockX = atoi(argv[7]); // block size in the i-direction
        blockY = atoi(argv[8]); // block size in the j-direction
    }
    else
    {
        printf("Usage: %s PO CFL L W NX NY n_iter block_x block_y\n", argv[0]);
        exit(1);
    }

    unsigned int tot_iters, R;
    REAL dx, dy, dt, kx, ky, t0, tFinal;

    dx = (REAL)L/(Nx-1); // dx, cell size
    dy = (REAL)W/(Ny-1); // dy, cell size
    dt = 1/(2*C*(1/dx/dx+1/dy/dy))*0.9; // dt, fix time step size
    kx = C/(12*dx*dx); // numerical conductivity
    ky = C/(12*dy*dy); // numerical conductivity
    t0 = 0.1; // Initial time
    tFinal = t0+dt*max_iters; //printf("Final time: %g\n",tFinal);
    R = 2; // halo regions width (in cells size)
    printf("dx: %g, dy: %g, dt: %g, final time: %g\n\n",dx,dy,dt,tFinal);

    // Copy constants to Constant Memory on the GPUs
    CopyToConstantMemory(kx, ky);

    // Initialize solution arrays
    REAL *h_u;
    
    h_u = (REAL*)malloc(sizeof(REAL)*(Nx*Ny));
    
    // Set Domain Initial Condition and BCs
    Call_Init2d(3,h_u,dx,dy,Nx,Ny);

    // GPU stream operations
    cudaStream_t compute_for_u;

    checkCuda(cudaStreamCreate(&compute_for_u));
    
    // GPU Memory Arrays
    size_t pitch_bites, pitch_gc_bites;

    REAL *d_u;
    REAL *d_uo;
    REAL *d_Lu;

    size_t pitch_bytes;
    checkCuda(cudaMallocPitch((void**)&d_u, &pitch_bytes,sizeof(REAL)*Nx,Ny));
    checkCuda(cudaMallocPitch((void**)&d_uo,&pitch_bytes,sizeof(REAL)*Nx,Ny));
    checkCuda(cudaMallocPitch((void**)&d_Lu,&pitch_bytes,sizeof(REAL)*Nx,Ny));
    unsigned int pitch = pitch_bytes/sizeof(REAL); printf("pitch: %d\n",pitch);

    // Copy Initial Condition from host to device
    checkCuda(cudaMemcpy2D(d_u,pitch_bytes,h_u,sizeof(REAL)*Nx,sizeof(REAL)*Nx,Ny,cudaMemcpyHostToDevice));

    // GPU kernel launch parameters
    dim3 threadsPerBlock1D(32,1,1); // initialization for C++
    unsigned int blocksInXY = DIVIDE_INTO(Nx*Ny,32); //getBlock(Nx, blockX);
    dim3 numBlocks1D(blocksInXY,1,1); // initialization for C++

    dim3 threadsPerBlock2D(blockX,blockY,1); // initialization for C++
    unsigned int blocksInX = DIVIDE_INTO(Nx,blockX); //getBlock(Nx, blockX);
    unsigned int blocksInY = DIVIDE_INTO(Ny,blockY); //getBlock(Nx, blockX);
    dim3 numBlocks2D(blocksInX,blocksInY,1); // initialization for C++

    // Initialized textures
    InitilizeTexture(d_u,Nx,Ny,pitch_bytes);

    // Set memory of temporal variables to zero
    checkCuda(cudaMemset2D(d_Lu,pitch_bytes,0,sizeof(REAL)*Nx,Ny));

    // Request the cpu current time
    time_t t = clock();
    
    // Call WENO-RK solver
    for(unsigned int iterations = 0; iterations < max_iters; iterations++)
    {
        // Runge Kutta Step 0
        checkCuda(cudaMemcpy2D(d_uo,pitch_bytes,d_u,pitch_bytes,sizeof(REAL)*Nx,Ny,cudaMemcpyDeviceToDevice));

        // Runge Kutta Step 1
        Call_Lu2d(numBlocks2D,threadsPerBlock2D,compute_for_u,pitch,Nx,Ny,d_Lu);
        Call_RK2d(numBlocks2D,threadsPerBlock2D,compute_for_u,1,pitch,Nx,Ny,dt,d_u,d_uo,d_Lu);
        
        // Runge Kutta Step 2
        Call_Lu2d(numBlocks2D,threadsPerBlock2D,compute_for_u,pitch,Nx,Ny,d_Lu);
        Call_RK2d(numBlocks2D,threadsPerBlock2D,compute_for_u,2,pitch,Nx,Ny,dt,d_u,d_uo,d_Lu);
        
        // Runge Kutta Step 3
        Call_Lu2d(numBlocks2D,threadsPerBlock2D,compute_for_u,pitch,Nx,Ny,d_Lu);
        Call_RK2d(numBlocks2D,threadsPerBlock2D,compute_for_u,3,pitch,Nx,Ny,dt,d_u,d_uo,d_Lu);
    }
    
    // Measure and Report computation time
    t += clock(); REAL tCPU = (REAL)t/CLOCKS_PER_SEC; printf("Computation took %lf seconds\n", tCPU);
    
    // Copy data from device to host
    checkCuda(cudaMemcpy2D(h_u,sizeof(REAL)*Nx,d_u,pitch_bytes,sizeof(REAL)*Nx,Ny,cudaMemcpyDeviceToHost));

    // uncomment to print solution to terminal
    if (DISPL) Print2D(h_u,Nx,Ny);

    float gflops = CalcGflops(tCPU, max_iters, Nx, Ny);
    PrintSummary("HeatEq2D (13-pt)", "none", tCPU, gflops, tFinal, max_iters, Nx, Ny);
    CalcError(h_u,tFinal,dx,dy,Nx,Ny);

    // Write solution to file
    if (WRITE) SaveBinary2D(h_u,Nx,Ny);
    
    // Free device memory
    checkCuda(cudaFree(d_u));
    checkCuda(cudaFree(d_uo));
    checkCuda(cudaFree(d_Lu));

    // Reset device
    checkCuda(cudaDeviceReset());

    // Free memory on host and device
    free(h_u);
    
    return 0;
}
