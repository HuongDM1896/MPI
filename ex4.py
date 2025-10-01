from mpi4py import MPI
import numpy as np

# Basic function to compute scalar product of two vectors
def scalarProduct(X, Y):
    return np.dot(X, Y)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Initialize X, Y, and N only on the root process (rank 0)
    
if rank == 0:
    N = 13# Example length, divisible by `size` 13 -> reduce 1 -> 12
    X = np.arange(1, N + 1, dtype='i')       # Vector [1, 2, ..., N]
    Y = np.arange(10, 10 * (N + 1), 10, dtype='i')  # Vector [10, 20, ..., 10*N]
    remainder = N % size
    if remainder != 0:
        X = X[:N - remainder]
        Y = Y[:N - remainder]
        N = len(X) 
    print(f"[Rank 0] Original vectors: X={X}, Y={Y}, N={N}")
else:
    X = None
    Y = None
    N = 0

# ================== Step 2: Broadcast N ===================
# Broadcast N to all ranks so everyone knows the size
N = comm.bcast(N, root=0)
chunk_size = N // size  # Each rank will handle this many elements

# ================== Step 3: Allocate local memory ===================
local_X = np.empty(chunk_size, dtype='i')
local_Y = np.empty(chunk_size, dtype='i')

# ================== Step 4: Scatter ===================
# Distribute chunks of X and Y to all ranks
comm.Scatter(X, local_X, root=0)
comm.Scatter(Y, local_Y, root=0)
print(f"[Rank {rank}] received local_X={local_X}, local_Y={local_Y}")

# ================== Step 5: Local computation ===================
# Each rank computes partial scalar product
local_result = scalarProduct(local_X, local_Y)

# ================== Step 6: Reduce ===================
# Collect partial results and sum them on root=0
total_result = comm.reduce(local_result, op=MPI.SUM, root=0)


# Step 7: Print the result on rank 0
if rank == 0:
    print("Total scalar product:", total_result)
    
# Collective choice: 
# comm.Scatter để chia vector X, Y cho các process.
# comm.reduce(..., op=MPI.SUM) để tổng hợp kết quả về rank 0.

# Memory management:
# np.empty(chunk_size, dtype='i') cho local_X và local_Y trên tất cả các rank.
# Root 0 giữ X, Y đầy đủ.

# Data balancing (khi N không chia hết size):
# Hiện tại code yêu cầu N % size == 0.
# Cắt bỏ phần dư (như bài pixel trước).
# Or Dùng Scatterv: cho phép chia vector với kích thước không đều
