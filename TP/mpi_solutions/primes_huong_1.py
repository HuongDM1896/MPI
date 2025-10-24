from mpi4py import MPI
import sys
import time
import numpy as np

def nb_primes(n):
    """
    Calculate the number of divisors for a given number n.
    
    Args:
        n (int): The number to find divisors for
    
    Returns:
        int: Count of how many numbers between 1 and n divide n evenly
    """
    result = 0
    for i in range(1, n+1):
        if n % i == 0:
            result += 1
    return result

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Current process ID (0 to size-1)
size = comm.Get_size()  # Total number of processes

# Get the upper bound N from command line argument
upper_bound = int(sys.argv[1])

############## BLOCK DISTRIBUTION APPROACH ################

# Synchronize all processes before timing starts
comm.barrier()
start_time = time.time()

# Calculate the work range for this process using BLOCK distribution
# Each process gets a contiguous block of numbers to process
a = rank * (upper_bound // size)      # Start of this process's block
b = (rank + 1) * (upper_bound // size) # End of this process's block

# Handle the case where upper_bound is not perfectly divisible by size
# The last process takes the remaining numbers
if rank == size - 1:
    b = upper_bound

# Debug print to show work distribution (commented out in production)
# print("Rank %d from %d to %d" % (rank, a,b))

# Initialize local maximum as a numpy array (required for MPI operations)
local_max = np.zeros(1, dtype='i')  # 'i' for integer type

# Process each number in this process's assigned block
# Note: range(a+1, b+1) because we process numbers from 1 to upper_bound
for val in range(a+1, b+1):
    tmp = nb_primes(val)              # Calculate number of divisors
    local_max[0] = max(local_max[0], tmp)  # Update local maximum

# Record the time after local computation (before MPI communication)
current_max = np.zeros(1, dtype='i')  # Buffer for receiving global result
local_end_time = time.time()

# Use MPI Reduce with MAX operation to find global maximum across all processes
# All processes send their local_max, root process receives the global maximum in current_max
comm.Reduce(local_max, current_max, root=0, op=MPI.MAX)

# Record total end time (including communication)
end_time = time.time()

# Print local computation time for each process to stderr
print('time local compute',' ', rank,':', local_end_time-start_time,'s', file=sys.stderr)

# Synchronize before final output
comm.barrier()

# Only root process prints the final result
if rank == 0:
    print('Block', current_max[0], 'in TOTAL TIME', end_time - start_time, 's')