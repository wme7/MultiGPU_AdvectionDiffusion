# profile heat3d
mpirun -np 2 nvprof -o test.%q{OMPI_COMM_WORLD_RANK}.nvprof ./heat3d_async.run 256 256 256 100 64 4 1