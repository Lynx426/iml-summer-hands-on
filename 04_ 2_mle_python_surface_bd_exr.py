import numpy as np
import matplotlib.pyplot as plt

# Generate data points
N = 50
S = np.arange(1, N, 0.1)  # Values from 1 to 49.9 in steps of 0.1
o = np.linspace(0.1, 0.9, 100)  # Values from 0.1 to 0.9 evenly spaced

# MLE 
def L(S, o):
    return S * np.log(o) + (N - S) * np.log(1. - o)

# Create a 1x2 subplot layout for plotting two graphs side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Maximum Likelihood Estimation")


# Compute the MLE values for all combinations of S and o and plot as a heatmap
heatmap = ax1.imshow(
    L(
        np.repeat(S[:, np.newaxis], len(o), axis=1),
        np.repeat(o[np.newaxis, :], len(S), axis=0)
    ),
    cmap='jet', origin='lower', aspect='auto', extent=[S.min(), S.max(), o.min(), o.max()]
)
ax1.set_xlabel('S')
ax1.set_ylabel('θ')
ax1.set_title("Bird's Eye View")

ax1.axvline(x=25, color='black')  # Add a vertical line at S=25

# L(o|S=25) Plot
# Plot the MLE values for a fixed S=25 against different values of o
ax2.plot(o, L(25, o), color='blue')
ax2.set_xlabel('o')
ax2.set_title("L(o|S=25)")

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Save the plot as an image
plt.savefig("./image/TM_s_theta_L_25")

# plt.show()
