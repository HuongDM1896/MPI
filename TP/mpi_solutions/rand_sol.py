# mpirun -n 2 python3 random_sol.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# try with and without commenting the next line
np.random.seed(0) # For reproducibility
numbers = np.random.randint(100, size=10, dtype='i')

n_max = np.empty(10, dtype='i')
n_min = np.empty(10, dtype='i')

comm.Reduce(numbers, n_max, op=MPI.MAX, root=0)
comm.Reduce(numbers, n_min, op=MPI.MIN, root=0)

if rank == 0:
    print('compare minmax', n_min, 'and', n_max)
    print(np.array_equal(n_min, n_max))
