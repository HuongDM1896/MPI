from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# We assume n and DIM are constant known by all processes
n = 30
DIM = 12

# Initialize matrix A only in the root process (rank 0)
if rank == 0:
    np.random.seed(10)

    A = np.random.rand(DIM,DIM)
    res = np.linalg.matrix_power(A, n)
    print('Signature ref', res.trace())

else:
    A = np.zeros((DIM,DIM))

# Broadcast the matrix A to all processes
comm.Bcast(A, root=0)

# Initialize the result matrix (same as A in the root process)
res = A.copy()

delta = rank*(DIM // size)

for _ in range(n-1):
    local_res = np.zeros((DIM//size, DIM))
    for lin in range(DIM//size):
        for col in range(DIM):
            for i in range(DIM):
                local_res[lin][col] += A[lin+delta][i] * res[i][col]
                
    # combine the result from all processes to the root process
    comm.Allgather(local_res, res)

# After all iterations, the result matrix `res` will hold A^n
# The root process calculates and prints the trace:
print('Signature mpi', res.trace())
