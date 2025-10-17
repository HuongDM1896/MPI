# mpirun -n 4 python3 stretching-base_sol_corrected.py

from mpi4py import MPI
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm

M = 255

# First method for stretching contrast
def f_one(x, n):
    if x == 0:
        return 0
    return int(M**(1 - n) * (x**n))

# Second method for stretching contrast
def f_two(x, n):
    if x == 0:
        return 0
    return int((M**((n - 1) / n)) * (x**(1 / n)))

# Converts an image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Loads an image on disk named "image.png" and converts it to grayscale
def readImage():
    img = mpimg.imread('image.png')
    plt.imshow(img)
    print("Press 'q' to continue")
    plt.show()
    grey = rgb2gray(img)
    pixels, nblines, nbcolumns = (np.ravel(grey) * 255).astype(np.int32), len(grey), len(grey[0])
    return pixels, nblines, nbcolumns

# Saves the image in "image-grey2-stretched.png" and shows it
def saveImage(newP, nblines, nbcolumns):
    newimg = newP.reshape((nblines, nbcolumns))
    plt.imshow(newimg, cmap=cm.Greys_r)
    print("Press 'q' to continue")
    plt.show()
    mpimg.imsave('image-grey2-stretched.png', newimg, cmap=cm.Greys_r)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Starting stretching...")
    pixels, nblines, nbcolumns = readImage()
    total_pixels = len(pixels)
else:
    pixels = None
    nblines = nbcolumns = total_pixels = 0

# Broadcast image dimensions to all processes
nblines = comm.bcast(nblines, root=0)
nbcolumns = comm.bcast(nbcolumns, root=0)
total_pixels = comm.bcast(total_pixels, root=0)

# Determine local chunk size for each process
local_nb = total_pixels // size
if rank == size - 1:
    # Last rank handles any remaining pixels
    local_nb += total_pixels % size

# Allocate array for local pixels
l_pixels = np.zeros(local_nb, dtype='i')

# Scatter pixel data
if rank == 0:
    # Reshape pixels to ensure it's contiguous and divisible by the number of processes
    padded_pixels = np.zeros(local_nb * size, dtype='i')
    padded_pixels[:total_pixels] = pixels  # Fill in actual pixel data
else:
    padded_pixels = None

# Scatter the padded pixels
comm.Scatter(padded_pixels, l_pixels, root=0)

# Compute min and max of local pixels
local_pix_min = np.array([np.min(l_pixels)], dtype='i')
local_pix_max = np.array([np.max(l_pixels)], dtype='i')

# Reduce to get global min and max across all processes
global_pix_min = np.zeros(1, dtype='i')
global_pix_max = np.zeros(1, dtype='i')

comm.Allreduce(local_pix_min, global_pix_min, op=MPI.MIN)
comm.Allreduce(local_pix_max, global_pix_max, op=MPI.MAX)

# Compute alpha, the parameter for contrast-stretching functions
alpha = 1 + (global_pix_max[0] - global_pix_min[0]) / M

# Apply contrast stretching in parallel using f_one or f_two
for i in range(len(l_pixels)):
    if rank % 2 == 0:
        l_pixels[i] = f_one(l_pixels[i], alpha)
    else:
        l_pixels[i] = f_two(l_pixels[i], alpha)

# Gather modified pixels at root process
if rank == 0:
    # Prepare to collect results
    new_pixels = np.empty(total_pixels, dtype='i')
else:
    new_pixels = None

comm.Gather(l_pixels[:total_pixels], new_pixels, root=0)

# Save the modified image at root
if rank == 0:
    saveImage(new_pixels, nblines, nbcolumns)
    print("Stretching done...")

