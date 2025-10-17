#! /usr/bin/python3

# mpirun -n 3 python3 monte_carlo_sol.py

from mpi4py import MPI
import time
import random
import numpy as np

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nb = 100001
    inside = np.zeros(1, dtype='i')

    # Set a unique random seed for each process
    random.seed(time.time() + rank)

    start_time = time.time()

    # Calculate the number of points each process will handle
    local_nb = nb // size
    if rank == size - 1:  # Last process does extra work if nb is not divisible by size
        local_nb += nb % size

    # Perform Monte Carlo simulation locally
    for _ in range(local_nb):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1:
            inside += 1

    # Use MPI Reduce to sum all inside counts to result at root
    result = np.zeros(1, dtype='i')
    comm.Reduce(inside, result, op=MPI.SUM, root=0)
    end_time = time.time()

    # Print process-specific information
    print(f"Process {rank} tried {local_nb} times, inside count: {inside[0]}")

    # Root process calculates and prints the result
    if rank == 0:
        pi_estimate = 4 * float(result[0]) / nb
        print(f"Pi ≈ {pi_estimate} (calculated in {end_time - start_time:.4f} seconds)")

