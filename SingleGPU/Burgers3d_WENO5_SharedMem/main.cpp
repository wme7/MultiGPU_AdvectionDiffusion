//
//  main.cpp
//  Burgers3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "Burgers.h"
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
int main(int argc, char** argv){
    
    REAL CFL, tEnd, L, W, H;
    unsigned int Nx, Ny, Nz;
    
    if (argc == 9)
    {
        tEnd= atof(argv[1]); // Final Time
        CFL= atof(argv[2]); // The stability parameter
        L = atof(argv[3]);  // domain lenght
        W = atof(argv[4]);  // domain width
        H = atof(argv[5]);  // domain height
        Nx = atoi(argv[6]); // number cells in x-direction
        Ny = atoi(argv[7]); // number cells in y-direction
        Nz = atoi(argv[8]); // number cells in z-direction
    }
    else
    {
        printf("Usage: %s tEnd CFL L W H NX NY NZ\n", argv[0]);
        exit(1);
    }

    // Define Constants
    REAL   dx = (REAL)L /(Nx-1);    // dx, cell size
    REAL   dy = (REAL)W /(Ny-1);    // dy, cell size
    REAL   dz = (REAL)H /(Nz-1);    // dy, cell size
    REAL   C = 0.00001;         // conduction coef
    REAL   kx = C/(12*dx*dx);   // dx, cell size
    REAL   ky = C/(12*dy*dy);   // dy, cell size
    REAL   kz = C/(12*dz*dz);   // dy, cell size
    printf("dx: %g, dy: %g, dz: %g, final time: %g\n\n",dx,dy,dz,tEnd);

    // Copy constants to Constant Memory on the GPUs
    CopyToConstantMemory(kx, ky, kz);

    // Initialize solution arrays
    REAL *h_u; checkCuda(cudaMallocHost((void**)&h_u, sizeof(REAL)*Nx*Ny*Nz));
   
    // Set Domain Initial Condition and BCs
    Call_Init3d(3,h_u,dx,dy,dz,Nx,Ny,Nz);

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
    clock_t start, stop;
    start= clock();
    checkCuda(cudaMemcpy2D(d_u, pitch_bytes, h_u, sizeof(REAL)*Nx, sizeof(REAL)*Nx, Ny*Nz, cudaMemcpyHostToDevice));
    stop = clock();
    double HtD_timer = (double)(stop-start)/CLOCKS_PER_SEC;

    // GPU kernel launch parameters*Ny*Nz
    dim3 threadsPerBlock3D(8,8,8); // initialization for C++
    unsigned int blocksInX = DIVIDE_INTO(Nx,8); //getBlock(Nx, blockX);
    unsigned int blocksInY = DIVIDE_INTO(Ny,8); //getBlock(Nx, blockX);
    unsigned int blocksInZ = DIVIDE_INTO(Nz,8); //getBlock(Nx, blockX);
    dim3 numBlocks3D(blocksInX,blocksInY,blocksInZ); // initialization for C++

    // Sweep texture in the x-direction
    dim3 threadsPerBlock_X(TILE+1,WIDTH,1);
    blocksInX = DIVIDE_INTO(Nx, TILE);
    blocksInY = DIVIDE_INTO(Ny,WIDTH);
    blocksInZ = DIVIDE_INTO(Nz,  1  );
    dim3 numBlocks_X(blocksInX,blocksInY,blocksInZ);

    // Sweep texture in the y-direction
    dim3 threadsPerBlock_Y(WIDTH,TILE+1,1);
    blocksInX = DIVIDE_INTO(Nx,WIDTH);
    blocksInY = DIVIDE_INTO(Ny, TILE);
    blocksInZ = DIVIDE_INTO(Nz,  1  );
    dim3 numBlocks_Y(blocksInX,blocksInY,blocksInZ);

    // Sweep texture in the z-direction
    dim3 threadsPerBlock_Z(WIDTH,TILE+1,1);
    blocksInX = DIVIDE_INTO(Nx,WIDTH);
    blocksInZ = DIVIDE_INTO(Nz, TILE);
    blocksInY = DIVIDE_INTO(Ny,  1  );
    dim3 numBlocks_Z(blocksInX,blocksInZ,blocksInY);

    // Synchronous Laplace
    // dim3 threadsPerBlock2D(8,8); 
    // blocksInX = DIVIDE_INTO(Nx,8); //getBlock(Nx, blockX);
    // blocksInY = DIVIDE_INTO(Ny,8); //getBlock(Nx, blockX);
    // dim3 numBlocks2D(blocksInX,blocksInY); 

    // Asynchronous Laplace
    dim3 threadsPerBlock2D(64,4,1);
    blocksInX = DIVIDE_INTO(Nx, 64 );
    blocksInY = DIVIDE_INTO(Ny,  4 );
    blocksInZ = DIVIDE_INTO(Nz,LOOP);
    dim3 numBlocks2D(blocksInX,blocksInY,blocksInZ);

    // Request the cpu current time
    start= clock();

    // Initialize time variables
    REAL dt =0; 
    REAL dt1=0; 
    REAL dt2=0; 
    REAL it =0;
    REAL  t =0;
    int step=0;
    checkCuda(cudaMemset2D(d_Lu, pitch_bytes, 0, sizeof(REAL)*Nx, Ny*Nz));
    
    // Call WENO-RK solver
    while (t < tEnd)
    {
        // Update/correct time step
        dt1 = CFL*dx/1.0; 
        dt2 = 1./(2*C*(1/dx/dx+1/dy/dy+1/dz/dz))*0.9;
        dt = dt1<dt2 ? dt1:dt2 ; 
        if ((t+dt)>tEnd){ dt=tEnd-t; } 
    
        // Update time and iteration counter
        t+=dt; it+=1;

        // Runge Kutta Step 0
        checkCuda(cudaMemcpy2D(d_uo, pitch_bytes, d_u, pitch_bytes, sizeof(REAL)*Nx, Ny*Nz, cudaMemcpyDeviceToDevice));
        // checkCuda(cudaMemset2D(d_Lu, pitch_bytes, 0, sizeof(REAL)*Nx, Ny*Nz));  // if Adv_x is not performed first

        // Runge Kutta Steps
        for(step=1; step<=3; step++) {
            Call_Adv_x(numBlocks_X,threadsPerBlock_X,compute_stream,pitch,Nx,Ny,Nz,dx,d_u,d_Lu);
            Call_Adv_y(numBlocks_Y,threadsPerBlock_Y,compute_stream,pitch,Nx,Ny,Nz,dy,d_u,d_Lu);
            Call_Adv_z(numBlocks_Z,threadsPerBlock_Z,compute_stream,pitch,Nx,Ny,Nz,dz,d_u,d_Lu);
            Call_Diff_(numBlocks2D,threadsPerBlock2D,compute_stream,pitch,Nx,Ny,Nz,d_u,d_Lu);
            Call_sspRK(numBlocks3D,threadsPerBlock3D,compute_stream,pitch,Nx,Ny,Nz,step,dt,d_u,d_uo,d_Lu);
        }
    }
    
    // Measure and Report computation time
    stop = clock(); double compute_timer = (double)(stop-start)/CLOCKS_PER_SEC; 
    printf("Computation took %1.3f seconds\n\n",compute_timer);
    printf("dt: %g, iterations: %g, final time: %g\n\n",dt,it,t);
    
    // Copy data from device to host
    start= clock();
    checkCuda(cudaMemcpy2D(h_u, sizeof(REAL)*Nx, d_u, pitch_bytes, sizeof(REAL)*Nx, Ny*Nz, cudaMemcpyDeviceToHost));
    stop = clock(); 
    double DtH_timer = (double)(stop-start)/CLOCKS_PER_SEC;

    // Write solution to file
    if (WRITE) SaveBinary3D(h_u,Nx,Ny,Nz);
    
    // Final Report
    float gflops = CalcGflops(compute_timer, it, Nx, Ny, Nz);
    PrintSummary("Burgers-3D GPU-WENO5", "Shared Memory", compute_timer, HtD_timer, DtH_timer, gflops, it, Nx, Ny, Nz);

    // Free device memory
    checkCuda(cudaFree(d_u ));
    checkCuda(cudaFree(d_uo));
    checkCuda(cudaFree(d_Lu));

    // Force Device to Reset
    checkCuda(cudaDeviceReset());

    // Free memory on host
    cudaFreeHost(h_u);
    
    return 0;
}
