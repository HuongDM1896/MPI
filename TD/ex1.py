from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # not use in this ex

# Create an array A with 2 elements initialized as zeros
A = np.zeros(2, dtype='f')
        
if rank == 0:
        A[1] = 42  # Only rank 0 sets the second element = 42
        # rank0: [0. 42.]
        # other ranks: [0. 0.]
print ("Before bcast", "A on ", rank, "=", A)	

# ================== Broadcast ===================
comm.Bcast(A, root=0) # Broadcast: copy data from root=0 to all other ranks
# After broadcast every rank has A = [0. 42.]
print ("After Bcast", "A on ", rank, "=", A)
# Each rank contributes its local A (2 elements)

# ================== Allgather ===================
result = np.empty(6, dtype = 'f')
comm.Allgather(A, result) # Allgather: collects all contributions and concatenates them
# For 3 ranks:
# result = [A(rank0), A(rank1), A(rank2)]
print ("After Allgather","result on", rank, "=", result)

A = A+1 # for numpy array, addition is element-wise
# Increase every element of A by 1
# Eg A = [0,42] → [1,43] on every rank
print("After addition, A on", rank, "=", A)

# ================== Gather ===================
comm.Gather(A, result, root=2)
# Gather: collect A from all ranks and store them in result on root=2
# Only root=2 has the final result, others keep old content in result
# Eg: rank0 A = [1,43], rank1 A = [1,43], rank2 A = [1,43]
# On root=2: result = [1,43, 1,43, 1,43]
print ("Gather", "A on", rank, "=", result )
