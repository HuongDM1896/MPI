# mpirun -n 4 python3 teams_sol.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    teams = np.random.randint(2, size=size, dtype='i')
    print(f"The file contains {teams}")
else:
    teams = None

my_team = np.empty(1, dtype='i')

comm.Scatter(teams, my_team, root=0)

colors = {0: 'blue', 1:'green'}
print('I am', rank, 'and my team is', colors[my_team[0]])
