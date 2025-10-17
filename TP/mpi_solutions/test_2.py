from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Constants
n = 30
DIM = 12

# Initialize matrix A and result matrix
if rank == 0:
    np.random.seed(10)
    A = np.random.rand(DIM, DIM)
    res = np.linalg.matrix_power(A, n)
    print('Signature ref', res.trace())
else:
    A = None

# Scatter rows of A to all processes
local_A = np.zeros((DIM // size, DIM))  # Each process receives a subset of rows
comm.Scatter(A, local_A, root=0)

# Prepare local result for multiplication
res = A.copy() if rank == 0 else np.zeros((DIM, DIM))

for _ in range(n-1):
    local_res = np.zeros((DIM // size, DIM))
    for lin in range(DIM // size):
        for col in range(DIM):
            for i in range(DIM):
                local_res[lin][col] += local_A[lin][i] * res[i][col]
    
    # Reduce the local results to rank 0
    comm.Reduce(local_res, res, op=MPI.SUM, root=0)
    
    # Broadcast the updated result matrix to all processes
    comm.Bcast(res, root=0)

if rank == 0:
    print('Signature mpi', res.trace())

