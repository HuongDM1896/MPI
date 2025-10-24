# mpirun -n 4 python3 primes_sol.py 40000

from mpi4py import MPI
import sys
import time
import numpy as np

def nb_primes(n):
    result = 0
    for i in range(1, n+1):
        if n%i == 0:
            result += 1
    return result

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

upper_bound = int(sys.argv[1])

############## BLOCK ################

comm.barrier()
start_time = time.time()

a = rank * (upper_bound//size)
b = (rank + 1) * (upper_bound//size)
if rank == size - 1:
    b = upper_bound
# print("Rank %d from %d to %d" % (rank, a,b))

local_max = np.zeros(1, dtype='i')
for val in range(a+1, b+1):
    tmp = nb_primes(val)
    local_max[0] = max(local_max[0], tmp)

current_max = np.zeros(1, dtype='i')
local_end_time = time.time()
comm.Reduce(local_max, current_max, root=0, op=MPI.MAX)

end_time = time.time()
print('time local compute',' ', rank,':', local_end_time-start_time,'s', file=sys.stderr)
comm.barrier()

if rank == 0:
    print('Block', current_max[0], 'in TOTAL TIME', end_time - start_time, 's')

################ ROUND ROBIN #####################

comm.barrier()
if rank == 0:
    print()
start_time = time.time()

local_max[0] = 0
for val in range(rank+1, upper_bound+1, size):
    tmp = nb_primes(val)
    local_max[0] = max(local_max[0], tmp)
local_end_time = time.time()

comm.Reduce(local_max, current_max, root=0, op=MPI.MAX)

end_time = time.time()

print(' ', rank,':', local_end_time-start_time,'s', file=sys.stderr)

comm.barrier()
if rank == 0:
    print('Round Robin', current_max[0], 'in', end_time - start_time, 's')
