# mpirun -n 4 python3 heat_huong.py
from mpi4py import MPI
import numpy as np
import random
import time
import matplotlib.pyplot as plt

def add_hot_spots(matrix, number):
    size_x, size_y = matrix.shape
    for i in range(number):
        x = random.randrange(1, size_x-1)
        y = random.randrange(1, size_y-1)
        matrix[x][y] = random.randint(500, 1000)

def get_val(matrix, x, y):
    tmp = matrix[x][y] + matrix[x-1][y] + matrix[x+1][y] + matrix[x][y-1] + matrix[x][y+1]
    return tmp // 5

def get_signature(matrix):
    return np.bitwise_xor.reduce(np.add.reduce(matrix))

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
width = 120
iterations = 20

# Determine rows per process (including ghost rows)
rows_per_proc = width // size
if rank < width % size:  # Handle uneven division
    rows_per_proc += 1

# Each process gets an extended submatrix (with ghost rows)
local_matrix = np.zeros((rows_per_proc + 2, width), dtype='i')  # +2 for ghost rows
tmp_matrix = np.zeros_like(local_matrix)

# Process 0 initializes the full matrix
if rank == 0:
    matrix = np.zeros((width, width), dtype='i')
    random.seed(3)
    add_hot_spots(matrix, 400)
else:
    matrix = None

# Scatter the matrix
sendcounts = [((width // size) + (1 if i < width % size else 0)) * width for i in range(size)]
displs = [sum(sendcounts[:i]) for i in range(size)]
flat_matrix = matrix.flatten() if rank == 0 else None

local_data = np.zeros(rows_per_proc * width, dtype='i')
comm.Scatterv([flat_matrix, sendcounts, displs, MPI.INT], local_data, root=0)
local_matrix[1:-1, :] = local_data.reshape((rows_per_proc, width))

# Timing
init_time = time.time()

# Heat propagation iterations
for _ in range(iterations):
    # Exchange ghost rows with neighbors
    if rank > 0:  # Send/receive from above
        comm.Sendrecv(local_matrix[1, :], dest=rank-1, recvbuf=local_matrix[0, :], source=rank-1)
    if rank < size - 1:  # Send/receive from below
        comm.Sendrecv(local_matrix[-2, :], dest=rank+1, recvbuf=local_matrix[-1, :], source=rank+1)

    # Compute new values for the inner rows
    for x in range(1, rows_per_proc + 1):
        for y in range(1, width-1):
            tmp_matrix[x][y] = get_val(local_matrix, x, y)

    # Swap matrices
    local_matrix[1:-1, :] = tmp_matrix[1:-1, :]

# Gather the submatrices
final_data = None
if rank == 0:
    final_data = np.zeros((width, width), dtype='i')

comm.Gatherv(local_matrix[1:-1, :].flatten(), [final_data, sendcounts, displs, MPI.INT], root=0)

final_time = time.time()

# Display results on process 0
if rank == 0:
    print('Total time:', final_time - init_time, 's')
    print('Signature:', get_signature(final_data))
    plt.imshow(final_data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig('heat_parallel.pdf')
    plt.show()

