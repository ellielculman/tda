"""
Extracting Features
@author: Dashti
"""

# Importing dependencies
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
from persim import plot_diagrams 

from PIL import Image
import matplotlib.pyplot as plt

def convert_gif_to_jpeg(gif_path, jpeg_path):
    try: 
        gif_image = Image.open(gif_path)
        gif_image.seek(0)
        single_frame = gif_image.convert("L")
        single_frame.save(jpeg_path, "JPEG")
    except Image.UnidentifiedImageError as e:
        print(f"Error: Unable to identify image file {gif_path}. Skipping.")
        print(e)


# Function to transfer image to LBP domain
def lbp8_image(img):
    n = np.size(img,0)
    m = np.size(img,1)
    padded_img = np.zeros((n+2, m+2))
    padded_img[1:n+1,1:m+1] = img
    lbp = np.zeros_like(img, dtype=int)
    
    row_idx = 0
    
    for i in range(1, n+1):
        col_idx = 0
        
        for j in range(1, m+1):
           bin_str = ''
           
           bin_str += '1' if (padded_img[i,j] > padded_img[i-1,j-1]) else '0'
           bin_str += '1' if (padded_img[i,j] > padded_img[i-1,j]) else '0'
           bin_str += '1' if (padded_img[i,j] > padded_img[i-1,j+1]) else '0'
           bin_str += '1' if (padded_img[i,j] > padded_img[i,j+1]) else '0'
           bin_str += '1' if (padded_img[i,j] > padded_img[i+1,j+1]) else '0'
           bin_str += '1' if (padded_img[i,j] > padded_img[i+1,j]) else '0'
           bin_str += '1' if (padded_img[i,j] > padded_img[i+1,j-1]) else '0'
           bin_str += '1' if (padded_img[i,j] > padded_img[i,j-1]) else '0'
           
           lbp[row_idx, col_idx] = int(bin_str, 2)
           col_idx += 1
            
        row_idx += 1
        
    return lbp




def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2)**2))

def create_distance_matrix(img):
    """
    Create a distance matrix for a given set of points using Euclidean distance.

    Parameters:
    - data: An array or list of points, where each point is represented as an array or list.

    Returns:
    - distance_matrix: A 2D numpy array representing the pairwise distances between points.
    """
    LBP = lbp8_image(img)
    n = np.size(LBP,0)
    m = np.size(LBP,1)
    max_dim = max(n,m)
    distance_matrix = np.zeros((max_dim, max_dim))

    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.linalg.norm(LBP[i] - LBP[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # Distance matrix is symmetric

    return distance_matrix


# Function to compute persistent diagrams
def get_pds(lbp_features, max_dim):
    feature_vectors = []
    
    # Create distance matrix
    dist_mat = create_distance_matrix(lbp_features)
    
    # Use only the non-zero part of the distance matrix
    non_zero_indices = np.nonzero(np.sum(dist_mat, axis=1))[0]
    dist_mat = dist_mat[non_zero_indices, :]
    dist_mat = dist_mat[:, non_zero_indices]
    
    if np.size(dist_mat) != 0:
        ripser_result = ripser(dist_mat, maxdim=max_dim, thresh=np.inf, distance_matrix=True)
        feature_vectors = ripser_result['dgms']
    
    return feature_vectors

# Main thread to run the script
if __name__ == '__main__':

    img_path = "images/Volume: benign_02 Case: A-1266-1/A_1266_1.LEFT_MLO.LJPEG.1_highpass.jpg"
    img = cv2.imread(img_path, 0)
    lbp_features = lbp8_image(img)
    max_dim = 1
    persistence_diagrams = get_pds(lbp_features, max_dim)
    X = np.load("persistence_diagram_dim1.npy")
    print(X)


'''
    root_directory = '/Users/pertsemlidish22/desktop/tda/images'
    file_extension = '*.gif'
    file_pattern = os.path.join(root_directory, file_extension)
    outputFolder = 'ExportedFeatures'
    count = 0
    for root, dirs, files in os.walk(root_directory):
        if root != root_directory:
            for filename in files:
                count += 1
                if "benign" in root:
                    type = "benign"
                if "cancer" in root or "Thumbnail" in root:
                    type = "cancer"
                if "normal" in root:
                    type = "normal"
                name = filename[0:14]
                if not os.path.isdir(f'{outputFolder}//{type}'):
                    os.mkdir(f'{outputFolder}//{type}//')
                if not os.path.isdir(f'{outputFolder}//{type}//{name}'):
                    os.mkdir(f'{outputFolder}//{type}//{name}')
                imgPath = os.path.join(root, filename)
                convert_gif_to_jpeg(imgPath, os.path.splitext(imgPath)[0] + ".jpg")
                imgPath_gif = os.path.splitext(imgPath)[0] + ".jpg"
                img= cv2.imread(imgPath_gif,0)
                barcode = get_pds(img, 1) 
                dim0 = barcode[0]
                dim1 = barcode[1]
                if not os.path.isdir(f'{outputFolder}//{type}//{name}'):
                    os.mkdir(f'{outputFolder}//{type}//{name}')
                np.save(f'{outputFolder}//{type}//{name}//dim0', dim0)
                np.save(f'{outputFolder}//{type}//{name}//dim1', dim1)



    X = np.load("ExportedFeatures/DDSM_Mass_257images/CSV/G0/R0/dim0_bin.npy" )
    Y = np.load("ExportedFeatures/DDSM_Mass_257images/CSV/G0/R1/dim0_bin.npy" )
    distanceXY = gudhi.wasserstein.wasserstein_distance(X, Y, matching=False, order=1.0, internal_p= 2.0, enable_autodiff=False, keep_essential_parts=True)
    print(distanceXY)
    '''