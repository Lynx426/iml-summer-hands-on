# import modules
import numpy as np
import matplotlib.pyplot as plt

# Set the number of trials (N) to 50
N = 50

# Create an array S with values from 1 to N (i.e., [1, 2, ..., 50])
S = np.arange(1, N + 1)

# Generate 100 equally spaced values between 0.1 and 0.9 to be used as theta
theta = np.linspace(0.1, 0.9, 100)

# Maximum Likelihood Estimation (MLE)
# Create a grid of (S, theta) pairs using meshgrid
S_grid, theta_grid = np.meshgrid(S, theta)

# Calculate the log-likelihood function L(theta|S) using MLE formula:
# L(theta|S) = S * log(theta) + (N - S) * log(1 - theta)
L = S_grid * np.log(theta_grid) + (N - S_grid) * np.log(1 - theta_grid)

# Create a 3D plot to visualize 
# Install 'qt' backend if required
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface using the calculated values of S, theta, and L
s = ax.plot_surface(S_grid, theta_grid, L, cmap='jet')

# Set labels for the three axes
ax.set_xlabel('S')
ax.set_ylabel('theta')
ax.set_zlabel('L(theta|S)')

#(elevation: 65 degrees, azimuth: 15 degrees)
ax.view_init(65, 15)

# Save the plot as an image in the "./pictures" directory with the name "TM_s_theta_L.png"
plt.savefig("./image/TM_s_theta_L.png")


