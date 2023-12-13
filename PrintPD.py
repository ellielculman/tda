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

from PIL import Image
import matplotlib.pyplot as plt 

def graph_pd():
    data = np.load("ExportedFeatures/normal/A_0005_1.RIGHT/dim1.npy")
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

if __name__ == '__main__':
    #graph_pd()
    X = np.load("ExportedFeatures/normal/A_0018_1.LEFT_/dim1.npy")
    print(X)
