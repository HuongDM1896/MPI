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

    start_time = time.time()
    #
    local_nb = nb // size
    if rank == size - 1: # Le dernier fait plus de travail
        local_nb += nb % size

    for _ in range(local_nb):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            inside +=1

    result = np.zeros(1, dtype='i')
    comm.Reduce(inside, result, op = MPI.SUM, root=0)
    #
    end_time = time.time()

    print("Process %s tried %s times" % (rank, local_nb))

    if rank == 0:
        print("Pi =", 4 * float(result[0])/nb, "in ", end_time-start_time, 'seconds')
