//
//  tools.cpp
//  diffusion2d-GPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "diffusion2d.h"

/*******************************/
/* Prints a flattened 2D array */
/*******************************/
void Print2D(REAL *u, unsigned int nx, unsigned int ny)
{
    unsigned int i, j;
    // print a single property on terminal
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            printf("%8.2f", u[i+nx*j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**************************/
/* Write to file 2D array */
/**************************/
void Save2D(REAL *u, unsigned int nx, unsigned int ny)
{
    unsigned int i, j;
    // print result to txt file
    FILE *pFile = fopen("result.txt", "w");
    if (pFile != NULL) {
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                //fprintf(pFile, "%d\t %d\t %g\n",j,i,u[i+nx*j]);
                fprintf(pFile, "%g\n",u[i+nx*j]);
            }
        }
        fclose(pFile);
    } else {
        printf("Unable to save to file\n");
    }
}

void SaveBinary2D(REAL *u, unsigned int nx, unsigned int ny)
{
    /* NOTE: We save our result as float values always!
     *
     * In Matlab, the results can be loaded by simply doing 
     * fID = fopen('result.bin');
     * result = fread(fID,[4,nx*ny],'float')';
     * myplot(result,nx,ny);
     */

    float data;
    unsigned int i, j, o;
    // print result to txt file
    FILE *pFile = fopen("result.bin", "w");
    if (pFile != NULL) {
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                o = i+nx*j; // index
                data = (float)u[o]; fwrite(&data,sizeof(float),1,pFile);
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
void Call_Init2d(int IC, REAL *u0, REAL dx, REAL dy, unsigned int nx, unsigned int ny)
{
    unsigned int i, j, o;
    
    switch (IC) {
        case 1: {
            // A square jump problem
            for (j= 0; j < ny; j++) {
                for (i= 0; i < nx; i++) {
                    o = i+nx*j;
                    if (i>0.4*nx && i<0.6*nx && j>0.4*ny && j<0.6*ny) {
                        u0[o]=1.0;
                    } else {
                        u0[o]=0.0;
                    }
                }
            }
            // Set Neumann boundary conditions in global domain u0'[0]=0.0;  u0'[NX]=0.0;
            break;
        }
        case 2: {
            // Set Everything to zero!
            for (j= 0; j < ny; j++) {
                for (i= 0; i < nx; i++) {
                    o = i+nx*j;
                    u0[o]=0.0;
                }
            }
            // Set Dirichlet boundary conditions in global domain u0[0]=0.0;  u0[NX]=0.0;
            break;
        }
        case 3: {
            for (j = 0; j < ny; j++) {
                for (i = 0; i < nx; i++) {
                    // set all domain's cells equal to : // exp( -(x^2+y^2)/(4*d*t0) )
                    o = i+nx*j;  u0[o] = EXP_DISTRIBUTION(i,j,dx,dy,1.0,0.1 ); 
                    // set BCs in the domain 
                    if (i==0 || i==nx-1 || j==0 || j==ny-1) u0[o] = 0.0;
                }
            }   
            // Set Dirichlet boundary conditions in global domain u0[0]=0.0;  u0[NX]=0.0;
            break;
        }
        // here to add another IC
    }
}

/******************/
/* COMPUTE GFLOPS */
/******************/
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny)
{
    return iterations*(double)((nx*ny) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/***********************/
/* COMPUTE ERROR NORMS */
/***********************/
void CalcError(REAL *u, REAL t, REAL dx, REAL dy, unsigned int nx, unsigned int ny)
{
  unsigned int i, j, o, xy;
  REAL err = 0., l1_norm = 0., l2_norm = 0., linf_norm = 0.;
 
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {

      //err = (exp(-2*M_PI*M_PI*t)*SINE_DISTRIBUTION(i,j,dx,dy)) - u[i+nx*j];
      err = ((0.1/t)*EXP_DISTRIBUTION(i,j,dx,dy,1.0,t)) - u[i];
      
      l1_norm += fabs(err);
      l2_norm += err*err;
      linf_norm = fmax(linf_norm,fabs(err));
    }
  }
  
  printf("L1 norm                                       :  %e\n", dx*dy*l1_norm);
  printf("L2 norm                                       :  %e\n", sqrt(dx*dy*l2_norm));
  printf("Linf norm                                     :  %e\n", linf_norm);
}

/*****************/
/* PRINT SUMMARY */
/*****************/
void PrintSummary(const char* kernelName, const char* optimization,
  REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds, 
  int computeIterations, int nx, int ny)
{
  printf("=========================== %s =======================\n", kernelName);
  printf("Optimization                                 :  %s\n", optimization);
  printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
  printf("===================================================================\n");
  printf("Total effective GFLOPs                       :  %lf\n", gflops);
  printf("===================================================================\n");
  printf("2D Grid Size                                 :  %d x %d \n", nx,ny);
  printf("Iterations                                   :  %d\n", computeIterations);
  printf("Final Time                                   :  %g\n", outputTimeInSeconds);
  printf("===================================================================\n");
}
