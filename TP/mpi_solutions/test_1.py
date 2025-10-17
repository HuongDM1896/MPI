import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpi4py import MPI

# Initializing MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Max intensity for grayscale
M = 255

# First contrast stretching function
def stretch_one(x, n):
    if x == 0:
        return 0
    return int(M ** (1 - n) * (x ** n))

# Second contrast stretching function
def stretch_two(x, n):
    if x == 0:
        return 0
    return int((M ** ((n - 1) / n)) * (x ** (1 / n)))

# Convert RGB image to grayscale
def convert_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Load the image, convert to grayscale, and display it
def load_image():
    img = mpimg.imread('image.png')
    plt.imshow(img)
    print("Press 'q' to continue")
    # plt.show()
    grey_image = convert_to_gray(img)
    pixels, nblines, nbcolumns = (np.ravel(grey_image) * 255).astype(np.int32), len(grey_image), len(grey_image[0])
    return pixels, nblines, nbcolumns

# Save the processed image and display it
def save_processed_image(new_pixels, nblines, nbcolumns):
    reshaped_image = new_pixels.reshape((nblines, nbcolumns))
    plt.imshow(reshaped_image, cmap=cm.Greys_r)
    print("Press 'q' to continue")
    plt.show()
    mpimg.imsave('image-grey2-stretched.png', reshaped_image, cmap=cm.Greys_r)

# Start processing, only process with rank 0 loads and processes the image
local_len = np.zeros(1, dtype='i')
if rank == 0:
    print("Starting contrast stretching...")
    pixels, nblines, nbcolumns = load_image()
    local_len[0] = len(pixels) // size
else:
    pixels = None

# Broadcast the local pixel length to all processes
comm.Bcast(local_len, root=0)
print(f"Process {rank} received len: {local_len[0]}")

# Initialize local pixels array before Scatter
local_pixels = np.zeros(local_len[0], dtype='i')  # Assuming size divides SIZE evenly
comm.Scatter(pixels, local_pixels, root=0)

print(f"Process {rank} received chunk:", local_pixels)

# Compute the minimum and maximum pixel values for contrast stretching
local_pix_min = min(local_pixels)
local_pix_max = max(local_pixels)

# Compute alpha, the parameter for contrast stretching functions
alpha = 1 + (local_pix_max - local_pix_min) / M

# Apply contrast stretching using the two different methods
if rank % 2 == 0:
    for i in range(len(local_pixels)):
        local_pixels[i] = stretch_one(local_pixels[i], alpha)
else:
    for i in range(len(local_pixels)):
        local_pixels[i] = stretch_two(local_pixels[i], alpha)

# Gather the processed pixels back to rank 0
comm.Gather(local_pixels, pixels, root=0)

# Rank 0 saves the final image after processing
if rank == 0:
    save_processed_image(pixels, nblines, nbcolumns)
    print("Contrast stretching completed...")


