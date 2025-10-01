# Exercise 1 - MPI Communication Example

The MPI (Message Passing Interface) functions use to demonstrate how data can be communicated between multiple processes. Below will explain what happens in the code when executed with three processes, including the expected output for each process, and provide a schema of the memory for each process.

---

## Elements of scripts 

1. **Initialization:**
   - `A = np.zeros(2, dtype='f')`: Initializes a NumPy array `A` of size 2, filled with zeros (floating-point numbers).

2. **Process Rank 0:**
   - If the rank is 0 (the root process), it sets the second element of `A` to 42 (`A[1] = 42`).

3. **Broadcast (`comm.Bcast`):**
   - `comm.Bcast(A, root=0)`: The root process (rank 0) broadcasts the array `A` to all other processes. After this operation, all processes will have the same value for `A`.

4. **Print Statement for Array `A`:**
   - Each process prints its rank and the contents of `A`.

5. **Allgather (`comm.Allgather`):**
   - `result = np.empty(6, dtype='f')`: Initializes a `result` array to hold the gathered data.
   - `comm.Allgather(A, result)`: Each process contributes its `A` to the `result` array, which will have a total size of 6 (2 elements from each of the 3 processes).

6. **Print Statement for `Allgather`:**
   - Each process prints its rank and the gathered `result`.

7. **Element-wise Addition:**
   - `A = A + 1`: Each process adds 1 to each element of its own array `A`.

8. **Gather (`comm.Gather`):**
   - `comm.Gather(A, result, root=2)`: Each process sends its updated `A` to the root process (rank 2). Only rank 2 will have the complete `result`.

9. **Print Statement for `Gather`:**
   - Each process prints its rank and the gathered `result`.

---

## Output

Assuming `rank` takes the values 0, 1, and 2 for the three processes, the output should be like this:


The final output from each process:

- `A on  0 = [ 0. 42.]`
- `AllGather 0 = [ 0. 42. 0. 42. 0. 42.]`
- `Gather 0 = [ 0. 42. 0. 42. 0. 42.]` (only prints for rank 2)

- `A on  1 = [ 0. 42.]`
- `AllGather 1 = [ 0. 42. 0. 42. 0. 42.]`
- `Gather 1 = [ 0. 42. 0. 42. 0. 42.]` (only prints for rank 2)

- `A on  2 = [ 0. 42.]`
- `AllGather 2 = [ 0. 42. 0. 42. 0. 42.]`
- `Gather 2 = [ 1. 43. 1. 43. 1. 43.]`

---

## Memory Schema

Below is the memory schema for each process, showing the variables `A`, `rank`, `result`, and `size`:

| Process | Variable | Memory Contents                |
|---------|----------|--------------------------------|
| **0**   | `A`      | `[0.0, 42.0]`                 |
|         | `rank`   | `0`                            |
|         | `result` | `[0.0, 42.0, 0.0, 42.0, 0.0, 42.0]` |
|         | `size`   | `6`                            |
| **1**   | `A`      | `[0.0, 42.0]`                 |
|         | `rank`   | `1`                            |
|         | `result` | `[0.0, 42.0, 0.0, 42.0, 0.0, 42.0]` |
|         | `size`   | `6`                            |
| **2**   | `A`      | `[0.0, 42.0]`                 |
|         | `rank`   | `2`                            |
|         | `result` | `[1.0, 43.0, 1.0, 43.0, 1.0, 43.0]` |
|         | `size`   | `6`                            |

---

## Summary

- Each process starts with `A` initialized to `[0.0, 0.0]`.
- After the `Bcast`, processes 1 and 2 have the same value of `A` as process 0, which is `[0.0, 42.0]`.
- The `Allgather` operation collects the values from all processes.
- The `Gather` operation results in the root process (rank 2) collecting the final modified values of `A`.

---

## Code: `ex1.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize A as a NumPy array of zeros
A = np.zeros(2, dtype='f')

# Process 0 sets the second element of A to 42
if rank == 0:
    A[1] = 42

# Broadcast A from root process (rank 0) to all processes
comm.Bcast(A, root=0)
print("A on ", rank, "=", A)

# Prepare result array for Allgather operation
result = np.empty(6, dtype='f')
comm.Allgather(A, result)
print("AllGather", rank, "=", result)

# Increment elements of A by 1 (element-wise)
A = A + 1

# Gather updated A arrays at root process (rank 2)
comm.Gather(A, result, root=2)
print("Gather", rank, "=", result)
```

# Exercise 2 - Scatter and Reduce Operations in MPI

The `Scatter` and `Reduce` operations in MPI. Break down each step and determine the output when the code is executed with 3 processes.

---

## Elements of scripts

1. **Initialization:**
   - For `rank == 0`, `A` is initialized as `np.arange(0, 6, dtype='f')`, resulting in an array: `[0., 1., 2., 3., 4., 5.]`.
   - For other ranks, `A` is set to `None`.

2. **Scatter (`comm.Scatter`):**
   - `comm.Scatter(A, local_A, root=0)`: The array `A` on process 0 is split equally and scattered across all processes. Each process receives a chunk of size 2 in `local_A`.
     - Process 0 receives `[0., 1.]`
     - Process 1 receives `[2., 3.]`
     - Process 2 receives `[4., 5.]`
   - Each process then prints its value of `local_A`.

3. **Element-wise Multiplication:**
   - `local_A = local_A * 3`: Each element in `local_A` is multiplied by 3, giving:
     - Process 0: `local_A = [0., 3.]`
     - Process 1: `local_A = [6., 9.]`
     - Process 2: `local_A = [12., 15.]`

4. **Reduce (`comm.Reduce` with `MPI.SUM`):**
   - `comm.Reduce(local_A, result, op=MPI.SUM, root=1)`: All processes send their `local_A` values to process 1, which computes the element-wise sum of these values and stores it in `result`.
     - Summing each element: `[0. + 6. + 12., 3. + 9. + 15.] = [18., 27.]`
   - Only process 1 will print the result for this reduction.

5. **Reduce (`comm.Reduce` with `MPI.MAX`):**
   - `comm.Reduce(local_A, result, op=MPI.MAX, root=0)`: All processes send their `local_A` values to process 0, which computes the element-wise maximum of these values and stores it in `result`.
     - Taking the maximum of each element: `[max(0., 6., 12.), max(3., 9., 15.)] = [12., 15.]`
   - Only process 0 will print the result for this reduction.

---

## Expected Output

The code will print the following:

- `local_A on 0 = [0. 1.]`
- `local_A on 1 = [2. 3.]`
- `local_A on 2 = [4. 5.]`

- `Reduced on 1 : rank = 1 result = [18. 27.]`
- `Reduced on 0 : rank = 0 result = [12. 15.]`

---

## Memory Schema

Then a memory schema for each process showing `A`, `local_A`, and `result`:

| Process | Variable  | Memory Contents                      |
|---------|-----------|--------------------------------------|
| **0**   | `A`       | `[0. 1. 2. 3. 4. 5.]`               |
|         | `local_A` | `[0. 3.]`                           |
|         | `result`  | `[12. 15.]` (after `MPI.MAX`)       |
| **1**   | `A`       | `None`                              |
|         | `local_A` | `[6. 9.]`                           |
|         | `result`  | `[18. 27.]` (after `MPI.SUM`)       |
| **2**   | `A`       | `None`                              |
|         | `local_A` | `[12. 15.]`                         |
|         | `result`  | Not modified (unused by `Reduce`)   |

---

## Summary of Actions

- **Scatter:** Distributes parts of `A` from process 0 to all processes.
- **Element-wise Multiply:** Each process multiplies its `local_A` elements by 3.
- **Reduce with `SUM`:** Calculates the sum of `local_A` across all processes and stores it in process 1.
- **Reduce with `MAX`:** Calculates the maximum of `local_A` across all processes and stores it in process 0.

---

## Code: `ex2.py`

```python
# mpi_example_ex2.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize A only in rank 0
if rank == 0:
    A = np.arange(0, 6, dtype='f')
else:
    A = None

# Each process receives a portion of A in local_A
local_A = np.zeros(2, dtype='f')
comm.Scatter(A, local_A, root=0)
print("local_A on rank", rank, "=", local_A)

# Multiply local_A by 3 element-wise
local_A = local_A * 3

# Perform a reduction with SUM on root=1
result = np.zeros(2, dtype='f')
comm.Reduce(local_A, result, op=MPI.SUM, root=1)
if rank == 1:
    print("Reduced on rank 1 (SUM): result =", result)

# Perform a reduction with MAX on root=0
comm.Reduce(local_A, result, op=MPI.MAX, root=0)
if rank == 0:
    print("Reduced on rank 0 (MAX): result =", result)
```

# Exercise 3 - Pixel Counting Using MPI Collectives

This code I use one photo then have to extract the pixels level, you can create a array of pixels instead.

---

## 1. Two Collectives Needed

To achieve the desired result (counting the number of pixels with values strictly over 100), we follow two collectives:

1. **Scatter**: Distributes portions of the `pixels` array from the root process (`rank == 0`) to all other processes. Each process will then have a chunk of pixels to analyze.
2. **Reduce**: Collects and sums the counts from each process (number of pixels with values over 100) back to a single process to get the final result.

---

## 2. Pixel Distribution Using Collectives

Assuming the number of pixels is divisible by the number of processes, we can distribute the `pixels` array using `Scatter`. Here’s how the pixel distribution and memory allocation should work:

- **Step 1**: On `rank == 0`, initialize the `pixels` array (e.g., with values from 0 to 255).
- **Step 2**: Use `np.empty` to allocate space for `local_pixels` on each process, which will hold the chunk of pixels each process receives.
- **Step 3**: Use `comm.Scatter` to distribute chunks of the `pixels` array from the root process to each process's `local_pixels` array.


### 3. Collective for Final Result

To obtain the final result (the total count of pixels with values over 100), each process needs to:

- **Step 1**: Count how many pixels in `local_pixels` are strictly greater than 100.
- **Step 2**: Use `comm.Reduce` with `MPI.SUM` to sum up all the counts across the processes, with the result being collected at the root process.


### Explanation of Memory Management

- **pixels**: Defined only on the root process (rank 0).
- **chunk_size**: Calculated on rank 0 and broadcast to all processes.
- **local_pixels**: Allocated with `np.empty` on each process to hold the scattered chunk.
- **local_count**: Each process calculates a local count of pixels > 100.
- **total_count**: Aggregated result at the root process after Reduce.

**Code Full**: 
```bash
mpirun -n 3 python3 ex3_correct_1.py

```python
from mpi4py import MPI
import numpy as np
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Load and preprocess the image only on the root process
if rank == 0:
    # Load the image
    image = Image.open("image.png").convert("L")  # Convert to grayscale
    pixels = np.array(image).flatten()  # Convert to 1D array of pixel values
    pixel_count = len(pixels)
    
    # Ensure the number of pixels is divisible by the number of processes
    if pixel_count % size != 0:
        raise ValueError("Number of pixels must be divisible by the number of processes.")
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

# Step 5: Each process counts its pixels with values > 100
local_count = np.sum(local_pixels > 100)

# Step 6: Reduce all local counts to get the total count at the root process
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Step 7: Only the root process outputs the final result
if rank == 0:
    print("Total number of pixels with value > 100:", total_count)

```

# Ex4: Computing the Scalar Product of Two Vectors in MPI

For this exercise, the goal is to compute the scalar product of two vectors X and Y, which are initially only available on rank 0. We'll distribute the computation across multiple processes in such a way that each process computes a portion of the scalar product, and then we gather the partial results to get the final scalar product.

## Step-by-Step

1. **Collectives Needed:**
   - **Scatter**: Distribute parts of the vectors X and Y from rank 0 to all processes.
   - **Reduce**: Sum the partial scalar products from each process to obtain the final result on rank 0.

2. **Part 1: Code Up to `scalarProduct` Function**  
   We need to:
   - Define X, Y, and N on rank 0.
   - Scatter portions of X and Y to each process.
   - Use the basic function `scalarProduct(X, Y)` to calculate the partial scalar product on each process.

3. **Part 2: Complete the Code to Print the Final Result on Rank 0**  
   After each process computes its partial scalar product, we use `Reduce` with `MPI.SUM` to gather and sum the partial products on rank 0. Rank 0 will then print the final result.

4. **Part 3: Removing the Divisibility Constraint**  
   To handle cases where N is not divisible by the number of processes, we’ll need to adjust the code so each process may receive a different number of elements. Let's break down the full solution accordingly.


```bash
python3 ex4.py
```

Or:

```bash
mpirun -n 1 python3 ex4.py
```

## Code: `ex4.py`

```python
from mpi4py import MPI
import numpy as np

# Basic function to compute scalar product of two vectors
def scalarProduct(X, Y):
    return np.dot(X, Y)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Initialize X, Y, and N only on the root process (rank 0)
if rank == 0:
    N = 12  # Example length, divisible by `size`
    X = np.arange(1, N + 1, dtype='i')       # Vector [1, 2, ..., N]
    Y = np.arange(10, 10 * (N + 1), 10, dtype='i')  # Vector [10, 20, ..., 10*N]
else:
    X = None
    Y = None
    N = 0

# Step 2: Broadcast the size of each chunk to all processes
N = comm.bcast(N, root=0)
chunk_size = N // size

# Step 3: Allocate local memory for chunks of X and Y
local_X = np.empty(chunk_size, dtype='i')
local_Y = np.empty(chunk_size, dtype='i')

# Step 4: Scatter parts of X and Y to all processes
comm.Scatter(X, local_X, root=0)
comm.Scatter(Y, local_Y, root=0)

# Step 5: Calculate partial scalar product on each process
local_result = scalarProduct(local_X, local_Y)

# Step 6: Reduce all partial results to get the final scalar product on rank 0
total_result = comm.reduce(local_result, op=MPI.SUM, root=0)

# Step 7: Print the result on rank 0
if rank == 0:
    print("Total scalar product:", total_result)
```

## Elements

- **Initialization**: Only rank 0 initializes X, Y, and N.
- **Broadcast**: The size N is broadcasted to all processes so they know the total number of elements.
- **Scatter**: X and Y are split into equal `chunk_size` parts and scattered to each process.
- **Local Computation**: Each process calculates the scalar product of its portion of X and Y using the `scalarProduct` function.
- **Reduce**: The partial results are summed on rank 0 using `MPI.SUM` to get the final scalar product.
- **Printing**: Only rank 0 prints the result.

## Removing the Constraint of N Divisible by the Number of Processes

To handle cases where N isn’t divisible by size, each process needs to receive a variable number of elements. Here’s how to modify the code:

1. **Determine Chunk Sizes**: Calculate `chunk_sizes` for each process, making sure to distribute the remaining elements evenly.
2. **Use Scatterv Instead of Scatter**: Use `Scatterv` to handle variable-sized chunks for each process.

## Modified Code for Non-Divisible N

```bash
python3 ex4_remove.py
```

## Code: `ex4_remove.py`

```python
from mpi4py import MPI
import numpy as np

# Basic function to compute scalar product of two vectors
def scalarProduct(X, Y):
    return np.dot(X, Y)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Initialize X, Y, and N only on the root process (rank 0)
if rank == 0:
    N = 13  # Example length, not divisible by `size`
    X = np.arange(1, N + 1, dtype='i')       # Vector [1, 2, ..., N]
    Y = np.arange(10, 10 * (N + 1), 10, dtype='i')  # Vector [10, 20, ..., 10*N]
else:
    X = None
    Y = None
    N = 0

# Step 2: Broadcast the size N to all processes
N = comm.bcast(N, root=0)

# Step 3: Calculate chunk sizes and offsets for each process
counts = [N // size + (1 if i < N % size else 0) for i in range(size)]
offsets = np.cumsum([0] + counts[:-1])

# Step 4: Allocate memory for each process's chunk of X and Y
local_X = np.empty(counts[rank], dtype='i')
local_Y = np.empty(counts[rank], dtype='i')

# Step 5: Scatter parts of X and Y to all processes with Scatterv
comm.Scatterv([X, counts, offsets, MPI.INT], local_X, root=0)
comm.Scatterv([Y, counts, offsets, MPI.INT], local_Y, root=0)

# Step 6: Calculate partial scalar product on each process
local_result = scalarProduct(local_X, local_Y)

# Step 7: Reduce all partial results to get the final scalar product on rank 0
total_result = comm.reduce(local_result, op=MPI.SUM, root=0)

# Step 8: Print the result on rank 0
if rank == 0:
    print("Total scalar product:", total_result)

```

## Explanation of Adjustments

- **Counts and Offsets**: `counts` determines how many elements each process should receive, while `offsets` helps specify where each chunk starts in X and Y.
- **Scatterv**: This method allows for variable-sized chunks, accommodating cases where N isn’t divisible by size.


# Ex5

## Target

1. **Calculate the mean of all elements in `vec` in parallel.**
2. **Determine how many elements are greater than the mean.**
3. **Ensure that only P0 displays the final count.**

## Steps by steps:

1. **Scatter**: Divide `vec` evenly across all processes using Scatter.
2. **Calculate Local Mean**: Each process calculates the mean of its subset.
3. **Global Mean Calculation**: Use `Reduce` with `MPI.SUM` to compute the global mean from all local means.
4. **Count Elements Greater Than Mean**: Each process counts the elements greater than the global mean in its subset.
5. **Reduce to Get Total Count**: Use `Reduce` to sum the counts of elements greater than the mean across all processes.
6. **Display Result**: Process P0 displays the final count.

## Solution Code

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Initialize vector on P0
if rank == 0:
    N = 12  # Number of elements, divisible by size
    vec = np.random.rand(N).astype('f') * 100  # Example vector with random float numbers
else:
    vec = None
    N = 0

# Step 2: Broadcast N to all processes
N = comm.bcast(N, root=0)
chunk_size = N // size

# Step 3: Scatter the vector to all processes
local_vec = np.empty(chunk_size, dtype='f')
comm.Scatter(vec, local_vec, root=0)

# Step 4: Each process calculates its local sum and count
local_sum = np.sum(local_vec)
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Step 5: Calculate the global mean on P0
if rank == 0:
    global_mean = global_sum / N
else:
    global_mean = None

# Broadcast the global mean to all processes
global_mean = comm.bcast(global_mean, root=0)

# Step 6: Each process counts elements greater than the global mean
local_count = np.sum(local_vec > global_mean)

# Step 7: Reduce all local counts to get the final count on P0
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Step 8: P0 prints the result
if rank == 0:
    print("Total number of elements greater than the mean:", total_count)
```

## Elements

1. **Initialize `vec` on rank 0**: Only process P0 initializes the vector `vec` with N random float numbers.
2. **Broadcast N**: N is broadcasted to ensure all processes are aware of the vector's length and can calculate their chunk sizes.
3. **Scatter**: The vector is divided equally among processes, with each receiving `chunk_size` elements into `local_vec`.
4. **Local and Global Sum Calculation**:
   - Each process calculates the sum of its local elements (`local_sum`).
   - `Reduce` with `MPI.SUM` computes `global_sum` on rank 0.
5. **Global Mean Calculation**:
   - Rank 0 computes the global mean (`global_mean = global_sum / N`) and broadcasts it to all processes.
6. **Counting Elements Greater Than the Mean**: Each process counts the elements in its `local_vec` greater than `global_mean`.
7. **Summing the Counts**: `Reduce` with `MPI.SUM` gathers the count of elements greater than the mean across all processes, storing the result on rank 0.
8. **Display**: Rank 0 prints the final result.

```bash
mpiexec -n 2 python3 ex5.py
```

