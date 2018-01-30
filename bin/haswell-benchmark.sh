#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J benchmark-haswell
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 1 -c 32 --cpu_bind=cores ./benchmark.py
