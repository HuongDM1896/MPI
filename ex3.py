from mpi4py import MPI
import numpy as np
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Load and preprocess the image only on the root process
if rank == 0:
    # Load the image only on root 0
    # image = Image.open("image.png").convert("L")  # Convert to grayscale
    # pixels = np.array(image).flatten()  # Convert to 1D array of pixel values [0..255]
    # use an array insteal of image
    pixels = np.array([ 12, 200, 5, 123, 240, 88, 50, 170, 210, 30, 199, 102], dtype='i')
    pixel_count = len(pixels)
    print("Number of pixels:", pixel_count)
    
    # Điều chỉnh: cut pixel if no devide to size
    remainder = pixel_count % size
    if remainder != 0:
        print(f"[Rank 0] Pixel count before adjustment: {pixel_count}")
        pixels = pixels[:pixel_count - remainder]  # bỏ đi phần dư
        pixel_count = len(pixels)
        print(f"[Rank 0] Pixel count after adjustment: {pixel_count}")
else:
    pixels = None
    pixel_count = 0

# Step 2: Broadcast pixel count and calculate chunk size
pixel_count = comm.bcast(pixel_count, root=0)
chunk_size = pixel_count // size

# Step 3: Allocate memory for each process's chunk
local_pixels = np.empty(chunk_size, dtype='i')

# Step 4: Scatter the pixels array from the root to all processes
comm.Scatter(pixels, local_pixels, root=0)
print(f"[Rank {rank}] received pixels {local_pixels}")

# Step 5: Each process counts its pixels with values > 100
local_count = np.sum(local_pixels > 100)
print(f"[Rank {rank}] local_count (pixels > 100) = {local_count}")

# Step 6: Reduce all local counts to get the total count at the root process
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Step 7: Only the root process outputs the final result
if rank == 0:
    print("Total number of pixels with value > 100:", total_count)
    
    
# Collectives needed: Scatter (to distribute pixels), Reduce (to sum counts).
# Memory management: np.empty(chunk_size) for scatter and reduce returns the result on root.