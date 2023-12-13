import gudhi
import numpy as np
import cv2
from scipy.spatial import distance
from ripser import ripser
import glob
import os
import multiprocessing as mp
from gudhi import representations 
from persim import PersistenceImager as pimgr
import gudhi.wasserstein
from ripser import Rips
from sklearn.metrics.pairwise import pairwise_distances


from PIL import Image
import matplotlib.pyplot as plt 

def graph_pd():
    '''
    data = np.load("ExportedFeatures/normal/A_0020_1.RIGHT/dim0.npy")
    print(data)
    data[np.isinf(data)] = np.nan  # Replace infinite values with NaN
    data[np.isnan(data)] = 1000000000  # Replace NaN with a large finite value

    rips = Rips()
    dgms = rips.fit_transform(data.T)

    # Plot the persistence diagram
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    rips.plot(data, show=False)
    plt.title("Point Cloud")

    plt.subplot(122)
    rips.plot(dgms, show=False)
    plt.title("Persistence Diagram")

    plt.show() 
    '''
    data = np.load("ExportedFeatures/normal/A_0005_1.LEFT_/dim0.npy")
    print(data)
    rips = Rips()
    diagrams = rips.fit_transform(data.T)

    # Plot the persistence diagram
    rips.plot(diagrams)
    plt.title('Example of Normal Mammogram Persistence Diagram')
    plt.show()
def graph_pd2():
    data = np.load("ExportedFeatures/normal/A_0020_1.RIGHT/dim0.npy")

    # Replace infinite values with NaN
    data[np.isinf(data)] = np.nan

    # Handle NaN values appropriately, for example, replace them with a large finite value
    data[np.isnan(data)] = 1000000000

    # Transpose the data for pairwise distance calculation
    data_transposed = data.T

    # Calculate pairwise distances
    distance_matrix = pairwise_distances(data_transposed, metric='euclidean')

    rips = Rips()
    dgms = rips.fit_transform(distance_matrix)

    # Plot the persistence diagram
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.scatter(data[:, 0], data[:, 1])  # Plot the point cloud
    plt.title("Point Cloud")

    plt.subplot(122)
    rips.plot(dgms, show=False)
    plt.title("Persistence Diagram")

    plt.show()


if __name__ == '__main__':
    #graph_pd()
    graph_pd2()
    #data = np.load("ExportedFeatures/normal/A_0020_1.RIGHT/dim0.npy")
    #print(data)
