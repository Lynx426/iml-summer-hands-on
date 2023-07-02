import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Install scipy and opencv-python via conda/pip if not already installed.

# Reading and preprocessing the input image
lambd = cv2.imread('venus2.tif')  # Reads the 'venus2.tif' image using OpenCV
lambd = cv2.cvtColor(lambd, cv2.COLOR_BGR2GRAY) / 255  # Converts the image to grayscale and normalizes pixel values to the range [0, 1]

T = 100  # Number of samples
lambdT = np.repeat(lambd[:, :, np.newaxis], T, axis=2)  # Creates a 3D array by repeating the grayscale image T times along the third dimension

# Generating Poisson-distributed samples
x = stats.poisson.rvs(lambdT)  # Generates Poisson-distributed random samples using the intensity values from lambdT

# Binarizing the samples
y = (x >= 1).astype(float)  # Converts the samples to binary values, where 1 indicates the presence of an event (intensity >= 1) and 0 indicates absence

# Maximum Likelihood Estimation (MLE)
lambdhat = -np.log(1 - np.mean(y, axis=2))  # Estimates the intensity values using MLE based on the binarized samples

# Displaying the estimated intensities
plt.imshow(lambdhat, cmap='gray')  # Displays the estimated intensities as a grayscale image
# plt.show()

# Saving the figure
plt.savefig('./image/TM_mle-venus2.png')  # Saves the figure as 'TM_mle-venus2.png' in the '/image' directory
