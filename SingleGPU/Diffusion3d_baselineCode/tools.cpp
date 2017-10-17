//
//  tools.cpp
//  diffusion3d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "acoustics3d.h"

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
    unsigned int i, j, k, xy;
    xy = nx*ny;
    // print result to txt file
    FILE *pFile = fopen("result.txt", "w");
    if (pFile != NULL) {
        for (k = 0; k < nz; k++) {
            for (j = 0; j < ny; j++) {
                for (i = 0; i < nx; i++) {
                    //fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,u[i+nx*j+xy*k]);
                    fprintf(pFile, "%g\n",u[i+nx*j+xy*k]);
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
     *  >> result = fread(fID,[4,nx*ny*nz],'float')';
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
void Call_Init3D(int IC, REAL *u0, REAL dx, REAL dy, REAL dz, 
  unsigned int nx, unsigned int ny, unsigned int nz)
{
  unsigned int i, j, k, o, xy; 
  xy=nx*ny;

  switch (IC) {
    case 1: {
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {
            // set all domain's cells equal to zero
            o = i+nx*j+xy*k;  u0[o] = 0.0;
            // set BCs in the domain 
            if (k==0)    u0[o] = 1.0; // bottom 
            if (k==1)    u0[o] = 1.0; // bottom
            if (k==nz-2) u0[o] = 1.0; // top
            if (k==nz-1) u0[o] = 1.0; // top
          }
        }
      }
      break;
    }
    case 2: {
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {
            // set all domain's cells equal to :
            o = i+nx*j+xy*k;  u0[o] = SINE_DISTRIBUTION(i,j,k,dx,dy,dz); 
            // set BCs in the domain 
            if (i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) u0[o] = 0.0;
          }
        }
      }
      break;
    }
    case 3: {
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {
            // set all domain's cells equal to : // exp( -(x^2+y^2+z^2)/(4*d*t0) )
            o = i+nx*j+xy*k;  u0[o] = EXP_DISTRIBUTION(i,j,k,dx,dy,dz,1.0,0.1); 
            // set BCs in the domain 
            if (i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) u0[o] = 0.0;
          }
        }
      }
      break;
    }
    // here to add another IC
  }
}

/******************/
/* COMPUTE GFLOPS */
/******************/
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    return iterations*(double)((nx*ny*nz) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/***********************/
/* COMPUTE ERROR NORMS */
/***********************/
void CalcError(REAL *u, REAL t, REAL dx, REAL dy, REAL dz, unsigned int nx, unsigned int ny, unsigned int nz)
{
  unsigned int i, j, k, o, xy;
  xy = nx*ny;

  REAL err = 0., l1_norm = 0., l2_norm = 0., linf_norm = 0.;

  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {

        //err = (exp(-3*M_PI*M_PI*t)*SINE_DISTRIBUTION(i,j,k,dx,dy,dz)) - u[i+nx*j+xy*k];
        err = (sqrt(0.1/t)*(0.1/t)*EXP_DISTRIBUTION(i,j,k,dx,dy,dz,1.0,t)) - u[i];
        
        l1_norm += fabs(err);
        l2_norm += err*err;
        linf_norm = fmax(linf_norm,fabs(err));
      }
    }
  }
  
  printf("L1 norm                                       :  %e\n", dx*dy*dz*l1_norm);
  printf("L2 norm                                       :  %e\n", sqrt(dx*dy*dz*l2_norm));
  printf("Linf norm                                     :  %e\n", linf_norm);
}

/*****************/
/* PRINT SUMMARY */
/*****************/
void PrintSummary(const char* kernelName, const char* optimization,
  REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds, 
  int computeIterations, int nx, int ny, int nz)
{
  printf("=========================== %s =======================\n", kernelName);
  printf("Optimization                                 :  %s\n", optimization);
  printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
  printf("===================================================================\n");
  printf("Total effective GFLOPs                       :  %lf\n", gflops);
  printf("===================================================================\n");
  printf("3D Grid Size                                 :  %d x %d x %d \n", nx,ny,nz);
  printf("Iterations                                   :  %d\n", computeIterations);
  printf("Final Time                                   :  %g\n", outputTimeInSeconds);
  printf("===================================================================\n");
}
