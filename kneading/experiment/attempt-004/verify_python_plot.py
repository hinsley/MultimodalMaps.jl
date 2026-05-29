import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("kneading/experiment/attempt-004/python_N150_ee_to_500.txt")
plt.figure()
plt.plot(data[:,0], data[:,1], 'b-o', markersize=3)
plt.xlim(0, 500)
plt.ylim(0.08, 0.16)
plt.savefig("kneading/experiment/attempt-004/verify_python_plot.png")
