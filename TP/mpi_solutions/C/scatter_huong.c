#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* data = NULL; // Dynamically allocate the array only on the root process

    if (rank == ROOT) {
        // Allocate memory for the data array based on the number of processes
        data = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            data[i] = 2 * i + 1; // Example data: 1, 3, 5, ...
        }
    }

    int recv_data;
    // Scatter the data array
    MPI_Scatter(data, 1, MPI_INT, &recv_data, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Print the received data
    printf("Process %d received data %d\n", rank, recv_data);

    // Free the dynamically allocated memory on the root process
    if (rank == ROOT) {
        free(data);
    }

    MPI_Finalize();
    return 0;
}
// cmd is: mpicc scatter_huong.c -o scatter_huong
// mpirun -n 6 ./scatter
