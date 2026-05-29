import numpy as np
np.random.seed(42)
N_points = 150
states = np.zeros((N_points, 12))
states[:, 0] = np.random.uniform(-10, 15, N_points)
states[:, 1] = np.random.uniform(-15, 10, N_points)
states[:, 2] = np.random.uniform(0, 30, N_points)
states[:, 3] = 1.0; states[:, 7] = 1.0; states[:, 11] = 1.0

np.savetxt("kneading/experiment/attempt-004/python_states.txt", states)
