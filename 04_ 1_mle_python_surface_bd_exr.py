import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
N = 50   # Total count
S = np.arange(1, N, 0.1)   # Range of observed counts
o = np.linspace(0.1, 0.9, 100)   # Range of parameter values

# MLE 
def L(S, o):
    # The likelihood function based on observed counts and parameter values
    return S * np.log(o) + (N - S) * np.log(1. - o)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("MaxI Likelihood Estimation")

# Bird's Eye View Heatmap
heatmap = ax1.imshow(
    L(
        np.repeat(S[:, np.newaxis], len(o), axis=1), 
        np.repeat(o[np.newaxis, :], len(S), axis=0)
    ),
    cmap='jet',
    origin='lower',
    aspect='auto',
    extent=[S.min(), S.max(), o.min(), o.max()]
)
ax1.set_xlabel('S')
ax1.set_ylabel('θ')
ax1.set_title("Bird's Eye View")

# Add a vertical line at S=12 on the first subplot
ax1.axvline(x=12, color='black')

# L(o|S=12) Plot
ax2.plot(o, L(12, o), color='blue')
ax2.set_xlabel('o')
ax2.set_title("L(o|S=12)")

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Save the figure as an image file
plt.savefig("./image/TM_s_theta_L_12.png")
