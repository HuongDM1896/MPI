# mpirun -n 2 python3 cumul_sol.py 1000

from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def cumul(a,b):
    return sum(range(a,b))

nb = int(sys.argv[1])   

local_nb = nb // size
local_a = local_nb*rank

l_res = np.array([cumul(local_a,local_a+local_nb)], dtype='i')

if rank==0:
    res = np.empty(1, dtype='i')
else:
    res = None
comm.Reduce(l_res, res, op=MPI.SUM, root=0)

if rank == 0:
    print(res[0])

