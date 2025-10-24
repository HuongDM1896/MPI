#! /usr/bin/python3
# mpirun -n 3 python3 heat_sol.py

import time
import random
import functools
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

def add_hot_spots(matrix, number):
    size_x, size_y = matrix.shape

    for i in range(number):
        x = random.randrange(1, size_x-1)
        y = random.randrange(1, size_y-1)
        matrix[x][y] = random.randint(500, 1000)

def get_val(matrix, x, y):
    tmp = matrix[x][y] + matrix[x-1][y] + matrix[x+1][y] + matrix[x][y-1] + matrix[x][y+1]
    return tmp // 5


def get_signature(matrix):
    return np.bitwise_xor.reduce(np.add.reduce(matrix))

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    width = 120

    if rank == 0:
        random.seed(3)
    
        matrix = np.zeros((width, width), dtype='i')
        add_hot_spots(matrix, 400)
    else:
        matrix = np.zeros((width, width), dtype='i')
        

    if rank == 0:
        init_time = time.time()

    comm.Bcast(matrix, root=0)
    
    a = rank * (width//size)
    if rank == 0:
        a = 1
    b = (rank + 1) * (width//size)
    if rank == size - 1:
        b = width-1
        
    tmp_matrix = np.zeros((width, width), dtype='i')

    for _ in range(20):
        for x in range(a, b):
            for y in range(1, width-1):
                tmp_matrix[x][y] = get_val(matrix, x, y)
        matrix, tmp_matrix = tmp_matrix, matrix

        if rank != size - 1:
            comm.Send(matrix[b-1], dest = rank+1)
        if rank != 0:
            comm.Recv(matrix[a-1], source = rank - 1)

        if rank != 0:
            comm.Send(matrix[a], dest = rank-1)
        if rank != size-1:
            comm.Recv(matrix[b], source = rank + 1)        

    if rank == 0:
        a=0
    if rank == size-1:
        b=width
    comm.Gather(matrix[a:b], matrix, root=0)

    if rank == 0:
        final_time = time.time()
        print('Total time:', final_time-init_time, 's')
        print('Signature:', get_signature(matrix))
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.savefig('heat.pdf')  
        plt.show(block=True)
