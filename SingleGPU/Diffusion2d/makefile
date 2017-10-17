# Coded by Manuel A. Diaz
# NHRI, 2016.06.10

# Compilers
NVCC = $(shell which nvcc)
CXX = $(shell which g++)

# CUDA paths
CUDA_INSTALL_PATH := /usr/local/cuda-7.5

# Device Architecture flags
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35
GENCODE_SM37 := -gencode arch=compute_37,code=sm_37
GENCODE_SM50 := -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS:= $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50)

# Compiler flags and libraries
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include
CFLAGS= -O3 -m64 -march=native -Wall -fopenmp -lgomp -funroll-loops
NVCCFLAGS = -O3 -m64 $(GENCODE_FLAGS)
# NVCCFLAGS = -O3 -m64 $(GENCODE_FLAGS) -Xcompiler -fopenmp -lgomp
CUDALIBFLAGS := -L $(CUDA_INSTALL_PATH)/lib64 -lcudart -lcuda

# Make instructions 
all: diffusion2d.run

diffusion2d.run: main.o tools.o kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $+ 

main.o: main.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

tools.o: tools.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

kernels.o: kernels.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -rf *.o *.run *.txt *.png *.vtk