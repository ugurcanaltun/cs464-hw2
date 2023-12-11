import numpy as np
import matplotlib.pyplot as plt
import gzip
import os

# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape((-1, 784))
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    return label_data

# Specifying a relative path to the local directory
dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, 'dataset')

images = read_pixels(filepath + '\\train-images-idx3-ubyte.gz')
labels = read_labels(filepath + '\\train-labels-idx1-ubyte.gz')

############################################################################################
# Question 1.1: Apply PCA on the dataset to obtain the principal components
# Report the proportion of variance explained (PVE) for the first 10 principal
# components and discuss your results

def apply_pca(data):
    # Center the data by subtracting the mean
    centered_data = data - np.mean(data, axis = 0)
    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_data, rowvar=False)
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]
    # Calculate the proportion of variance explained (PVE)
    total_variance = np.sum(np.abs(eigenvalues))
    pve_calculated = np.cumsum(np.abs(eigenvalues)) / total_variance
    return eigenvectors, pve_calculated

# Get the first 10 principal components and PVE
NUM_COMPONENTS = 10
principal_components, pve = apply_pca(images)

# Report PVE for the first 10 principal components
print("Proportion of Variance Explained (PVE) for the first 10 principal components:")
print(pve[:NUM_COMPONENTS])

############################################################################################
# Question 1.2: Report at least how many of the principal components should be used to
# explain the 70% of the data.
REQUIRED_PVE = 0.7
num_components_70 = np.argmax(pve >= REQUIRED_PVE) + 1
print(f"Number of principal components to explain 70% of the data: {num_components_70}")

############################################################################################
# Question 1.3: Using the first 10 principal components found in Question 1.1, reshape
# each principal component to a 28 x 28 matrix. Apply min-max scaling to each principal
# component to set the range of values to [0,1] so that the principal components can be
# visualized. After scaling, display the obtained greyscale principal component images
# of size 28 x 28. Discuss your results.

def visualize_principal_components(principal_components, num_components):
    for i in range(num_components):
        # Reshape to 28x28
        pc_image = np.real(principal_components[:,i]).reshape((28, 28))
        # Apply min-max scaling
        pc_image = (pc_image - np.min(pc_image)) / (np.max(pc_image) - np.min(pc_image))
        # Display the principal component
        plt.subplot(2, 5, i + 1)
        plt.imshow(pc_image, cmap='Greys_r')
        plt.title(f"PC {i + 1}")
        plt.axis('off')
    plt.show()

# Display the first 10 principal components
visualize_principal_components(principal_components, NUM_COMPONENTS)

############################################################################################
# Question 1.4: Project the first 100 images of the dataset onto the
# first 2 principal components. Plot the projected data points on the
# 2-D space by coloring them according to the labels provided in the
# dataset. Label the axes by the index of their corresponding
# principal components. Each digit label should be colored with a
# different color, 10 colors in total. Discuss the distribution of the
# data points according to their labels by considering the visuals of
# the first 2 principal component found in Question 1.3.

def plot_pca_projection(data, labels, principal_components, num_components):
    # Project data onto the first 2 principal components
    projected_data = np.dot(data, np.real(principal_components[:,:num_components]))

    # Plot the data points
    plt.figure(figsize=(8, 6))
    for digit in range(10):
        indices = labels == digit
        plt.scatter(projected_data[indices, 0], projected_data[indices, 1], label=str(digit))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('Projection of the first 100 images onto the first 2 principal components')
    plt.show()

# Project the first 100 images
plot_pca_projection(images[:100], labels[:100], principal_components, NUM_COMPONENTS)

############################################################################################
# Question 1.5: Describe how you can reconstruct an original digit image
# using the principal components found in Question 1.1. Use first k
# principal components to analyze and reconstruct the first image in
# the dataset where k ∈ {1, 50, 100, 250, 500, 784}. Discuss your results.

def reconstruct_image(original, principal_components, num_components):
    # Project the original image onto the selected principal components
    projected_image = np.dot(original, np.real(principal_components[:, :num_components]))

    # Reconstruct the image by adding back the mean
    reconstructed_image = np.dot(projected_image, np.real(principal_components[:, :num_components]).T) + np.mean(original, axis=0)

    return reconstructed_image

plt.figure(figsize=(15,3))

# Reconstruct the first image using different numbers of principal components
k_values = [1, 50, 100, 250, 500, 784]
for i,k in enumerate(k_values, 1):
    reconstructed = reconstruct_image(images[0], principal_components, k)

    # Reshape and display the reconstructed image
    reconstructed_image = reconstructed.reshape((28, 28))
    plt.subplot(1, len(k_values), i)
    plt.imshow(reconstructed_image, cmap='Greys_r')
    plt.title(f'{k} Principal Components')
    plt.axis('off')

plt.tight_layout()
plt.show()
    