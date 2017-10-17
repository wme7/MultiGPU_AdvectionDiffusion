//
//  tools.cpp
//  Burgers3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "Burgers.h"

/*******************************/
/* Prints a flattened 3D array */
/*******************************/
void Print3D(REAL *u, unsigned int nx, unsigned int ny, unsigned int nz)
{
    unsigned int i, j, k, xy;
    xy=nx*ny;
    // print a single property on terminal
    for(k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                printf("%8.2f", u[i+nx*j+xy*k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

/*****************************/
/* Write ASCII file 3D array */
/*****************************/
void Save3D(REAL *u, unsigned int nx, unsigned int ny, unsigned int nz)
{
    unsigned int i, j, k, xy, o;
    xy = nx*ny;
    // print result to txt file
    FILE *pFile = fopen("result.txt", "w");
    if (pFile != NULL) {
        for (k = 0; k < nz; k++) {
            for (j = 0; j < ny; j++) {
                for (i = 0; i < nx; i++) {
                    o = i+nx*j+xy*k; // index
                    //fprintf(pFile, "%d\t %d\t %d\t %g\t %g\t %g\t %g\n",k,j,i,u[o],v[o],w[o],p[o]);
                    fprintf(pFile, "%g\n",u[o]);
                }
            }
        }
        fclose(pFile);
    } else {
        printf("Unable to save to file\n");
    }
}

/******************************/
/* Write Binary file 3D array */
/******************************/
void SaveBinary3D(REAL *u, unsigned int nx, unsigned int ny, unsigned int nz)
{
    /* NOTE: We save our result as float values always!
     *
     * In Matlab, the results can be loaded by simply doing 
     *  >> fID = fopen('result.bin');
     *  >> result = fread(fID,[1,nx*ny*nz],'float')';
     *  >> myplot(result,nx,ny,nz);
     */

    float data;
    unsigned int i, j, k, xy, o;
    xy = nx*ny;
    // print result to txt file
    FILE *pFile = fopen("result.bin", "w");
    if (pFile != NULL) {
        for (k = 0; k < nz; k++) {
            for (j = 0; j < ny; j++) {
                for (i = 0; i < nx; i++) {
                    o = i+nx*j+xy*k; // index
                    data = (float)u[o]; fwrite(&data,sizeof(float),1,pFile);
                }
            }
        }
        fclose(pFile);
    } else {
        printf("Unable to save to file\n");
    }
}

/***************************/
/* PRESSURE INITIALIZATION */
/***************************/
void Call_Init3d(int IC, REAL *u0, REAL dx, REAL dy, REAL dz, unsigned int nx, unsigned int ny, unsigned int nz)
{
  unsigned int i, j, k, xy, o;
  xy = nx*ny;
  switch (IC) {
    case 1: {
      // A square jump problem
      for (k= 0; k < nz; k++) {
        for (j= 0; j < ny; j++) {
          for (i= 0; i < nx; i++) {
            o = i+nx*j+xy*k;
            if (i>0.4*nx && i<0.6*nx && j>0.4*ny && j<0.6*ny && k>0.4*nz && k<0.6*nz) {
                u0[o]=1.0;
            } else {
                u0[o]=0.0;
            }
          }
        }
      }
      // Set Neumann boundary conditions in global domain u0'[0]=0.0;  u0'[NX]=0.0;
      break;
    }
    case 2: {
      // Homogeneous IC
      for (k= 0; k < nz; k++) {
        for (j= 0; j < ny; j++) {
          for (i= 0; i < nx; i++) {
            o = i+nx*j+xy*k;
            u0[o]=0.0;
          }
        }
      }
      break;
    }
    case 3: {
      // Homogeneous IC
      for (k= 0; k < nz; k++) {
        for (j= 0; j < ny; j++) {
          for (i= 0; i < nx; i++) {
            o = i+nx*j+xy*k;
            u0[o]=GAUSSIAN_DISTRIBUTION(i*dx,j*dy,k*dz);
          }
        }
      }
      break;
    }
    // here to add another IC
  }
}

/********************/
/* Calculate Gflops */
/********************/
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    return (3*iterations)*(double)((nx * ny * nz) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/****************************/
/* Print Experiment Summary */
/****************************/
void PrintSummary(const char* kernelName, const char* optimization,
    double computeTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, 
    float gflops, int computeIterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    printf("===========================%s=======================\n", kernelName);
    printf("Optimization                                 :  %s\n", optimization);
    printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
    printf("Data transfer(s) HtD                         :  %lf seconds\n", hostToDeviceTimeInSeconds);
    printf("Data transfer(s) DtH                         :  %lf seconds\n", deviceToHostTimeInSeconds);
    printf("===================================================================\n");
    printf("Total effective GFLOPs                       :  %lf\n", gflops);
    printf("===================================================================\n");
    printf("3D Grid Size                                 :  %d x %d x %d\n",nx,ny,nz);
    printf("Iterations                                   :  %d x 3 RK steps\n", computeIterations);
    printf("===================================================================\n");
}