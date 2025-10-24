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

################ ROUND ROBIN / CYCLIC DISTRIBUTION #####################

# Synchronize all processes before timing starts
comm.barrier()
if rank == 0:
    print()  # Empty line for better output formatting
start_time = time.time()

# Initialize local maximum as a numpy array (required for MPI operations)
local_max = np.zeros(1, dtype='i')  # 'i' for integer type
local_max[0] = 0  # Initialize to 0

# ROUND ROBIN DISTRIBUTION: Each process handles numbers with stride = size
# Process 0: 1, 1+size, 1+2*size, 1+3*size, ...
# Process 1: 2, 2+size, 2+2*size, 2+3*size, ...
# Process 2: 3, 3+size, 3+2*size, 3+3*size, ...
# ...
# Process k: k+1, k+1+size, k+1+2*size, ...
for val in range(rank + 1, upper_bound + 1, size):
    tmp = nb_primes(val)              # Calculate number of divisors for current number
    local_max[0] = max(local_max[0], tmp)  # Update local maximum

# Record the time after local computation (before MPI communication)
local_end_time = time.time()

# Buffer for receiving global result
current_max = np.zeros(1, dtype='i')

# Use MPI Reduce with MAX operation to find global maximum across all processes
# All processes send their local_max, root process receives the global maximum in current_max
comm.Reduce(local_max, current_max, root=0, op=MPI.MAX)

# Record total end time (including communication)
end_time = time.time()

# Print local computation time for each process to stderr
print('time local compute', ' ', rank, ':', local_end_time - start_time, 's', file=sys.stderr)

# Synchronize before final output
comm.barrier()

# Only root process prints the final result
if rank == 0:
    print('Round Robin', current_max[0], 'in', end_time - start_time, 's')