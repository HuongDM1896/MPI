from mpi4py import MPI
import numpy as np

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # not use in this ex

if rank == 0:
	A = np.arange(0, 6, dtype='f') # A = [0. 1. 2. 3. 4. 5.]
else:
	A = None
	
local_A = np.zeros(2, dtype = 'f') # temp memory start as 0.0 of
comm.Scatter(A, local_A, root=0) 
# Divide array A into equal sized blocks and send each block to each process.
# Eg, 3 processes -> 6 elements /3 for each process
# rank 0 → [0,1], rank 1 → [2,3], rank 2 → [4,5]
print ("local_A on ", rank, "=", local_A)

# multi *3 each element on local_A
local_A = local_A * 3 # numpy : the multiplication is element-wise
print("After multiply , local_A on rank", rank, "=", local_A)
# rank 0: [0,1] → [0,3]
# rank 1: [2,3] → [6,9]
# rank 2: [4,5] → [12,15]

# ================== Reduce SUM ===================
result = np.zeros(2, dtype='f') # Array of result on root
comm.Reduce(local_A, result, op=MPI.SUM, root=1) 
# Reduce: sum (MPI.SUM) data from all ranks to root=1
print ('Reduced on 1 : rank =', rank, 'result = ', result)
# rank1 gets the cumulative result: 
# [0+6+12, 3+9+15] = [18, 27]

# ================== Reduce MAX ===================
comm.Reduce(local_A, result, op=MPI.MAX, root=0)
# rank0 gets the max result of each element:
# [max(0,6,12), max(3,9,15)] = [12, 15]
print ('Reduced on 0 : rank =', rank, 'result = ', result)

