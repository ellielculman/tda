import gudhi
from gudhi import plot_persistence_barcode
import numpy as np
import cv2
from scipy.spatial import distance
from ripser import ripser
import glob
import os
import multiprocessing as mp
from gudhi import representations 
from persim import plot_diagrams
import gudhi.wasserstein
from ripser import Rips
from sklearn.metrics.pairwise import pairwise_distances


from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib 
import ExportFeatures

def graph_pd():
    max_dim = 1
    imgPath = "images/Volume: cancer_08 Case: A-1533-1/A_1533_1.RIGHT_MLO.LJPEG.1_highpass.gif"
    ExportFeatures.convert_gif_to_jpeg(imgPath, os.path.splitext(imgPath)[0] + ".jpg")
    imgPath_gif = os.path.splitext(imgPath)[0] + ".jpg"
    img= cv2.imread(imgPath_gif,0)
    lbp_features = ExportFeatures.lbp8_image(img)
    persistence_diagrams = ExportFeatures.get_pds(lbp_features, max_dim)

    fig, ax = plt.subplots()

    # Visualize the persistence diagram
    plot_diagrams(persistence_diagrams, show=True, ax=ax)
    plt.xlabel("Birth Time")
    plt.ylabel("Death Time")
    plt.title('Normal Mammogram Persistence Diagram')
    plt.show()




if __name__ == '__main__':
    #graph_pd()
    #graph_pd()
    #data = np.load("ExportedFeatures/normal/A_0020_1.RIGHT/dim0.npy")
    #graph_barcode()
    #print(data)
    
    max_dim = 1
    imgPath = "images/Volume: cancer_08 Case: A-1533-1/A_1533_1.RIGHT_MLO.LJPEG.1_highpass.gif"
    ExportFeatures.convert_gif_to_jpeg(imgPath, os.path.splitext(imgPath)[0] + ".jpg")
    imgPath_gif = os.path.splitext(imgPath)[0] + ".jpg"
    img= cv2.imread(imgPath_gif,0)
    print(img)
    print(ExportFeatures.lbp8_image(img))
    ''''
    imgPath_gif = os.path.splitext(imgPath)[0] + ".jpg"
    img= cv2.imread(imgPath_gif,0)
    lbp_features = ExportFeatures.lbp8_image(img)
    persistence_diagrams = ExportFeatures.get_pds(lbp_features, max_dim)

    gudhi.plot_persistence_barcode(persistence_diagrams[1], persistence_file='', alpha=0.6, max_intervals=20000, inf_delta=0.1, legend=None, colormap=None, axes=None, fontsize=16)
    plt.title("Cancerous Dimension 1 Barcode")
    plt.show()
    '''


    