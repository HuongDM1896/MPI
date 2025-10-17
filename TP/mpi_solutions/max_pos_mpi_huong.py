# max_pos_mpi.py
from mpi4py import MPI
import numpy as np

def get_local_max(tab_chunk):
    """Finds the local maximum value and its position in a chunk of the array."""
    local_pos = np.argmax(tab_chunk)
    local_max = tab_chunk[local_pos]
    return local_max, local_pos

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only the root process generates the array
    SIZE = 12
    if rank == 0:
        np.random.seed(42)
        tab = np.random.randint(100, size=SIZE, dtype='i')
        print(f"Array: {tab}")
    else:
        tab = None
    
    # Determine the chunk size for each process
    chunk_size = SIZE // size
    if SIZE % size != 0:
        chunk_size += 1  # handle uneven distribution
    
    # Each process will receive a chunk of the array
    tab_chunk = np.empty(chunk_size, dtype='i')
    comm.Scatter(tab, tab_chunk, root=0)
    
    # Each process computes the local maximum and its position
    local_max, local_pos = get_local_max(tab_chunk)
    
    # Gather all local maximums and positions at the root
    all_local_max = comm.gather(local_max, root=0)
    all_local_pos = comm.gather(local_pos, root=0)
    
    if rank == 0:
        # Compute the global maximum from the local maximums
        global_max = max(all_local_max)
        global_max_index = all_local_max.index(global_max)
        
        # Calculate the global position of the max element
        global_position = global_max_index * chunk_size + all_local_pos[global_max_index]
        
        print(f"The maximum value is {global_max} at position {global_position}")

if __name__ == "__main__":
    main()

