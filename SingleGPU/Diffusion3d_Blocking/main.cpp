//
//  main.cpp
//  diffusion3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "acoustics3d.h"
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
    unsigned int L, W, H, Nx, Ny, Nz, max_iters;

    if (argc == 9)
    {
        C = atof(argv[1]); // conductivity, here it is assumed: Cx = Cy = Cz = C.
        L = atoi(argv[2]); // domain lenght
        W = atoi(argv[3]); // domain width 
        H = atoi(argv[4]); // domain height
        Nx = atoi(argv[5]); // number cells in x-direction
        Ny = atoi(argv[6]); // number cells in y-direction
        Nz = atoi(argv[7]); // number cells in z-direction
        max_iters = atoi(argv[8]); // number of iterations / time steps
    }
    else
    {
        printf("Usage: %s diffCoef L W H NX NY NZ n_iter\n", argv[0]);
        exit(1);
    }

    unsigned int tot_iters, R;
    REAL dx, dy, dz, dt, kx, ky, kz, t0, tFinal;

    dx = (REAL)L/(Nx-1); // dx, cell size
    dy = (REAL)W/(Ny-1); // dy, cell size
    dz = (REAL)H/(Nz-1); // dz, cell size
    dt = 1/(2*C*(1/dx/dx+1/dy/dy+1/dz/dz))*0.9; // dt, fix time step size
    kx = C/(12*dx*dx); // numerical conductivity
    ky = C/(12*dy*dy); // numerical conductivity
    kz = C/(12*dz*dz); // numerical conductivity
    t0 = 0.1; // Initial time
    tFinal = t0+dt*max_iters; //printf("Final time: %g\n",tFinal);
    R = 2; // halo regions width (in cells size)
    printf("dx: %g, dy: %g, dz: %g, dt: %g, final time: %g\n\n",dx,dy,dz,dt,tFinal);   

    // Copy constants to Constant Memory on the GPUs
    CopyToConstantMemory(kx, ky, kz);

    // Initialize solution arrays
    REAL *h_u; h_u = (REAL*)malloc(sizeof(REAL)*Nx*Ny*Nz);
    
    // Set Domain Initial Condition and BCs
    Call_Init3D(3,h_u,dx,dy,dz,Nx,Ny,Nz);

    // GPU stream operations
    cudaStream_t compute_stream;
    checkCuda(cudaStreamCreate(&compute_stream));
   
    // GPU Memory Arrays
    size_t pitch_bytes;
    REAL *d_u;  checkCuda(cudaMallocPitch((void**)&d_u , &pitch_bytes, sizeof(REAL)*Nx, Ny*Nz));
    REAL *d_uo; checkCuda(cudaMallocPitch((void**)&d_uo, &pitch_bytes, sizeof(REAL)*Nx, Ny*Nz));
    REAL *d_Lu; checkCuda(cudaMallocPitch((void**)&d_Lu, &pitch_bytes, sizeof(REAL)*Nx, Ny*Nz));
    unsigned int pitch = pitch_bytes/sizeof(REAL); printf("pitch: %d\n",pitch);

    // Copy Initial Condition from host to device
    checkCuda(cudaMemcpy2D(d_u, pitch_bytes, h_u, sizeof(REAL)*Nx, sizeof(REAL)*Nx, Ny*Nz, cudaMemcpyDefault));

    // GPU kernel launch parameters*Ny*Nz
    dim3 threadsPerBlock3D(32,4,4); // initialization for C++
    unsigned int blocksInX = DIVIDE_INTO(Nx,32); //getBlock(Nx, blockX);
    unsigned int blocksInY = DIVIDE_INTO(Ny, 4); //getBlock(Nx, blockX);
    unsigned int blocksInZ = DIVIDE_INTO(Nz, 4); //getBlock(Nx, blockX);
    dim3 numBlocks3D(blocksInX,blocksInY,blocksInZ); // initialization for C++

    dim3 threadsPerBlock2D(32,4,1); // initialization for C++
    blocksInX = DIVIDE_INTO(Nx,32); //getBlock(Nx, blockX);
    blocksInY = DIVIDE_INTO(Ny,4); //getBlock(Nx, blockY);
    blocksInZ = DIVIDE_INTO(Nz,k_loop); //getBlock(Nx, blockZ);
    dim3 numBlocks2D(blocksInX,blocksInY,blocksInZ); // initialization for C++

    // Set memory of temporal variables to zero
    checkCuda(cudaMemset2D(d_Lu, pitch_bytes, 0, sizeof(REAL)*Nx, Ny*Nz));

    // Request the cpu current time
    time_t t = clock();
    
    // Call WENO-RK solver
    for(unsigned int iterations = 0; iterations < max_iters; iterations++)
    {
        // Runge Kutta Step 0
        checkCuda(cudaMemcpy2D(d_uo, pitch_bytes, d_u, pitch_bytes, sizeof(REAL)*Nx, Ny*Nz, cudaMemcpyDefault));

        // Runge Kutta Step 1
        Call_Lu3d(numBlocks2D,threadsPerBlock2D,compute_stream,pitch,Nx,Ny,Nz,d_u,d_Lu);
        Call_RK3d(numBlocks3D,threadsPerBlock3D,compute_stream,1,pitch,Nx,Ny,Nz,dt,d_u,d_uo,d_Lu);
        
        // Runge Kutta Step 2
        Call_Lu3d(numBlocks2D,threadsPerBlock2D,compute_stream,pitch,Nx,Ny,Nz,d_u,d_Lu);
        Call_RK3d(numBlocks3D,threadsPerBlock3D,compute_stream,2,pitch,Nx,Ny,Nz,dt,d_u,d_uo,d_Lu);
        
        // Runge Kutta Step 3
        Call_Lu3d(numBlocks2D,threadsPerBlock2D,compute_stream,pitch,Nx,Ny,Nz,d_u,d_Lu);
        Call_RK3d(numBlocks3D,threadsPerBlock3D,compute_stream,3,pitch,Nx,Ny,Nz,dt,d_u,d_uo,d_Lu);
    }
    
    // Measure and Report computation time
    t += clock(); REAL tCPU = (REAL)t/CLOCKS_PER_SEC; printf("Computation took %lf seconds\n", tCPU);
    
    // Copy data from device to host
    checkCuda(cudaMemcpy2D(h_u, sizeof(REAL)*Nx, d_u, pitch_bytes, sizeof(REAL)*Nx, Ny*Nz, cudaMemcpyDeviceToHost));

    // uncomment to print solution to terminal
    if (DISPL) Print3D(h_u,Nx,Ny,Nz);

    float gflops = CalcGflops(tCPU, max_iters, Nx, Ny, Nz);
    PrintSummary("HeatEq3D (13-pt)", "PitchMem + Blocking", tCPU, gflops, tFinal, max_iters, Nx, Ny, Nz);
    CalcError(h_u, tFinal, dx, dy, dz, Nx, Ny, Nz);

    // Write solution to file
    if (WRITE) SaveBinary3D(h_u,Nx,Ny,Nz);
    
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
