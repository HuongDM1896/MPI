# mpirun -n 2 python3 number_sol.py 42

from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

number = np.empty(1, dtype='i')
if rank==0:
    number[0] = int(sys.argv[1])

comm.Bcast(number, root=0)

print("From process of rank", rank, "the passnumber is", number)


