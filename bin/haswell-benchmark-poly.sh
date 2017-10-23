#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -J benchmark-haswell
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
for p in 1 2 4 8 16 32 64; do
    OMP_NUM_THREADS=$p srun -n 1 -c $p --cpu_bind=cores ./benchmark-poly.py
done
