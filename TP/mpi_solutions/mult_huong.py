from mpi4py import MPI
import numpy as np

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants for matrix size and power (n is the power, DIM is the matrix dimension)
n = 30
DIM = 12

# Initialize matrix A only in the root process (rank 0)
if rank == 0:
    np.random.seed(10)
    A = np.random.rand(DIM, DIM)
else:
    A = None

# Broadcast the matrix A to all processes
A = comm.bcast(A, root=0)

# Initialize the result matrix (same as A in the root process)
res = np.copy(A)

# Compute the matrix power iteratively
for _ in range(n - 1):
    # Create a temporary matrix for storing the product of A and res
    tmp = np.zeros((DIM, DIM))
    
    # Each process calculates a chunk of the matrix multiplication
    for lin in range(rank * DIM // size, (rank + 1) * DIM // size):
        for col in range(DIM):
            for i in range(DIM):
                tmp[lin][col] += A[lin][i] * res[i][col]

    # Gather the result from all processes to the root process
    comm.Allreduce(tmp, res, op=MPI.SUM)

# After all iterations, the result matrix `res` will hold A^n
# The root process calculates and prints the trace
#if rank == 0:
print('Signature    ', np.trace(res))

