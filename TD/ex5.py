# Scatter vector, cal local sum (also global sum cause only rank 0 in root) 
# → Reduce để tính mean 
# → đếm số phần tử > mean 
# → Reduce để tổng về P0.

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ================== Step 1: Initialize vector on root ===================
if rank == 0:
    # Example vector of 12 floats
    vec = np.array([1.2, 5.6, 7.8, 3.3, 12.1, 8.7, 0.5, 6.6, 9.0, 4.4, 11.0, 2.2], dtype='f')
    N = len(vec) # 12 elements
    print(f"[Rank 0] Original vector: {vec}")
else:
    vec = None
    N = 0

# ================== Step 2: Broadcast N ===================
N = comm.bcast(N, root=0)
chunk_size = N // size

# ================== Step 3,4: Allocate local memory & Scatter vector ===================
local_vec = np.empty(chunk_size, dtype='f')
comm.Scatter(vec, local_vec, root=0)
print(f"[Rank {rank}] received local_vec = {local_vec}")

# ================== Step 5: Local sum for mean ===================
local_sum = np.sum(local_vec)
# Reduce sum to compute total sum at root
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Root computes the mean
if rank == 0:
    mean_value = total_sum / N
    print(f"[Rank 0] Mean value = {mean_value}")
else:
    mean_value = None

# Broadcast mean to all ranks
mean_value = comm.bcast(mean_value, root=0)

# ================== Step 6: Local count of elements > mean ===================
local_count = np.sum(local_vec > mean_value)
print(f"[Rank {rank}] local_count of elements > mean = {local_count}")

# ================== Step 7: Reduce to total count at root ===================
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# ================== Step 8: Output result on root ===================
if rank == 0:
    print(f"Total number of elements greater than mean: {total_count}")
