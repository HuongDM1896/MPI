# mpirun -n 4 python3 stretching-base_sol.py

from mpi4py import MPI
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import time

M = 255

# First method for stretching contrast
def f_one(x,n):
        if x==0:
                return 0
        return int(M**(1-n) * (x**n))

# Second method for stretching contrast
def f_two(x,n):
        if x==0:
                return 0
        return int((M**((n-1)/n)) * (x**(1/n)))

# Converts an image to grayscale
def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Loads an image on disk named "image.png" and convert it to greyscale, and shows it
def readImage():
        img = mpimg.imread('image.png')
        plt.imshow(img)
        print("Press 'q' to continue")
        plt.show()
        grey = rgb2gray(img)
        pixels, nblines, nbcolumns = (np.ravel(grey)*255).astype(np.int32), len(grey), len(grey[0])
        return pixels, nblines, nbcolumns

# Saves the image in "image-grey2-stretched.png" and shows it
def saveImage(newP, nblines, nbcolumns):
        newimg = newP.reshape((nblines, nbcolumns))
        plt.imshow(newimg, cmap = cm.Greys_r)
        print("Press 'q' to continue")
        plt.show()
        mpimg.imsave('image-grey2-stretched.png', newimg, cmap = cm.Greys_r )

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start_time = time.time()

if rank == 0:
        print("Starting stretching...")
        pixels, nblines, nbcolumns = readImage()
else:
        pixels = None

l_nb = np.zeros(1, dtype='i')
if rank == 0:
        l_nb[0] = len(pixels)//size
comm.Bcast(l_nb, root=0)

l_pixels = np.zeros(l_nb[0], dtype='i')

comm.Scatter(pixels, l_pixels, root=0)

# compute min and max of local pixels
local_pix_min = np.array([min(l_pixels)], dtype = 'i')
local_pix_max = np.array([max(l_pixels)], dtype = 'i')

global_pix_min = np.zeros(1, dtype='i')
global_pix_max = np.zeros(1, dtype='i')

comm.Allreduce(local_pix_min, global_pix_min, MPI.MIN)
comm.Allreduce(local_pix_max, global_pix_max, MPI.MAX)

# compute alpha, the parameter for f_* functions
alpha = 1+(global_pix_max - global_pix_min) / M

# stretch contrast for all pixels. f_one and f_two are the two different methods
for i in range(0,len(l_pixels)):
        if rank % 2 == 0:
                l_pixels[i] = f_one(l_pixels[i], alpha)
        else:
	        l_pixels[i] = f_two(l_pixels[i], alpha)

comm.Gather(l_pixels, pixels, root=0)
end_time = time.time()
if rank == 0:
        # save the image
        saveImage(pixels, nblines, nbcolumns)
        print("Stretching done...")
        print({end_time - start_time})


