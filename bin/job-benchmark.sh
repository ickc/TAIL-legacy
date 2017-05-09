#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl,quad,flat
#SBATCH -p debug
#SBATCH -J benchmark tail
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=272
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 1 -c 272 --cpu_bind=cores numactl -p 1 ./benchmark.py
