//
//  Tools.c
//  Burgers3d-GPU-MPI
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "BurgersMPICUDA.h"

/*******************************/
/* Prints a flattened 2D array */
/*******************************/
void Print2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
  unsigned int i, j;
  // print a single property on terminal
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      printf("%g ", u[i+nx*j]);
    }
    printf("\n");
  }
  printf("\n");
}

/*****************************/
/* Write ASCII file 3D array */
/*****************************/
void Save_2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
  unsigned int i, j;
  // print result to txt file
  FILE *pFile = fopen("result.bin", "w");
  if (pFile != NULL) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        fprintf(pFile, "%g\n",u[i+nx*j]);
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
void SaveBinary2D(REAL *u, const unsigned int nx, const unsigned int ny, const char *name)
{
  /* NOTE: We save our result as float values always!
   *
   * In Matlab, the results can be loaded by simply doing 
   *  >> fID = fopen('result.bin');
   *  >> result = fread(fID,[1,nx*ny*nz],'float')';
   *  >> myplot(result,nx,ny,nz);
   */

  float data;
  unsigned int i, j, o;
  // print result to txt file
  FILE *pFile = fopen(name, "w");
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

/**********************/
/* Initializes arrays */
/**********************/
void Init_domain(const int IC, REAL *u0, const REAL dx, const REAL dy, const unsigned int nx, const unsigned int ny)
{
	unsigned int i, j, o;
	switch (IC) {
    case 1: {
      // A Square Jump problem
      for (j= 0; j < ny; j++) {
        for (i= 0; i < nx; i++) {
          o = i+nx*j;
          if (i>=nx/4 && i<3*nx/4 && j>=ny/4 && j<3*ny/4) {
            u0[o]=1.;
          } else {
            u0[o]=0.;
          }
        }
      }
      break;
    }
    case 2: {
      // Homogeneous IC
      for (j= 0; j < ny; j++) {
        for (i= 0; i < nx; i++) {
          o = i+nx*j;
          u0[o]=0.0;
        }
      }
      break;
    }
		case 3: {
			// Sine Distribution in pressure field
			for (j = 0; j < ny; j++) {
				for (i = 0; i < nx; i++) {
					o = i+nx*j; 
					if (i==0 || i==nx-1 || j==0 || j==ny-1) {
						u0[o] = 0.0;
					} else {
						u0[o] = GAUSSIAN_DISTRIBUTION((0.5*(nx-1)-i)*dx,(0.5*(ny-1)-j)*dy);
					}
				}
			}
			break;
		}
		// Here to add another IC
	}
}

/******************************/
/* Initialize the sub-domains */
/******************************/
void Init_subdomain(REAL *h_q, REAL *h_s_q, const unsigned int n, const unsigned int Nx, const unsigned int _Ny)
{
	unsigned int idx_2d; // Global 3D index
	unsigned int idx_sd; // Subdomain index
	unsigned int i, j;

	// Copy Domain into n-subdomains
	for (j = 0; j < _Ny+2*RADIUS; j++) {
		for (i = 0; i < Nx; i++) {

			idx_2d = i+Nx*(j+n*_Ny);
			idx_sd = i+Nx*(j);

			h_s_q[idx_sd] = h_q[idx_2d];
		}
	}
}

/*******************************************************/
/* Merges the smaller sub-domains into a larger domain */
/*******************************************************/
void Merge_domains(REAL *h_s_q, REAL *h_q, const unsigned int n, const unsigned int Nx, const unsigned int _Ny)
{
	unsigned int idx_2d; // Global 3D index
	unsigned int idx_sd; // Subdomain index
	unsigned int i, j;

	// Copy n-subdomains into the Domain
	for (j = RADIUS; j < _Ny+RADIUS; j++) {
		for (i = 0; i < Nx; i++) {

			idx_2d = i+Nx*(j+n*_Ny);
			idx_sd = i+Nx*(j);

			h_q[idx_2d] = h_s_q[idx_sd];
		}
	}
}

/******************************/
/* Function to initialize MPI */
/******************************/
void InitializeMPI(int* argc, char*** argv, int* rank, int* numberOfProcesses)
{
	MPI_CHECK(MPI_Init(argc, argv));
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, rank));
	MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, numberOfProcesses));
	MPI_CHECK(MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
}

/****************************/
/* Function to finalize MPI */
/****************************/
void FinalizeMPI()
{
	MPI_CHECK(MPI_Finalize());
}

/********************/
/* Calculate Gflops */
/********************/
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny)
{
    return (3*iterations)*(double)((nx * ny) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/****************************/
/* Print Experiment Summary */
/****************************/
void PrintSummary(const char* kernelName, const char* optimization,
    double computeTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, 
    float gflops, const int computeIterations, unsigned int nx, unsigned int ny)
{
    printf("===========================%s=======================\n", kernelName);
    printf("Optimization                                 :  %s\n", optimization);
    printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
    printf("Data transfer(s) HtD                         :  %lf seconds\n", hostToDeviceTimeInSeconds);
    printf("Data transfer(s) DtH                         :  %lf seconds\n", deviceToHostTimeInSeconds);
    printf("===================================================================\n");
    printf("Total effective GFLOPs                       :  %lf\n", gflops);
    printf("===================================================================\n");
    printf("3D Grid Size                                 :  %d x %d\n",nx,ny);
    printf("Iterations                                   :  %d x 3 RK steps\n", computeIterations);
    printf("===================================================================\n");
}