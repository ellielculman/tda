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

from PIL import Image
import matplotlib.pyplot as plt

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

# Function to create distance matrix
def ulbp_top_basis(img, lbp_idx):
    LBP = lbp8_image(img)
    binindex = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48,
                56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131,
                135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227,
                231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
    
    n = np.size(LBP,0)
    m = np.size(LBP,1)
    pts = []
    dist_mat = []

    for i in range(n): 
        for j in range(m): 
            if(LBP[i,j] == binindex[lbp_idx - 1]):
                pts.append([i,j])
    
    if(np.size(pts) > 0):
        dist_mat = distance.squareform(distance.pdist(pts))
    
    return dist_mat

# Function to get all roatations of a geometry
def get_geomtetry_ulbp(geometry_no):
    
    geometry_mat = [[2, 3, 5, 8, 12, 17, 23, 30],
                    [4, 6, 9, 13, 18, 24, 31, 37],
                    [7, 10, 14, 19, 25, 32, 38, 43],
                    [11, 15, 20, 26, 33, 39, 44, 48],
                    [16, 21, 27, 34, 40, 45, 49, 52],
                    [22, 28, 35, 41, 46, 50, 53, 55],
                    [29, 36, 42, 47, 51, 54, 56, 57]]
    
    if(geometry_no in range(7)):
        return geometry_mat[geometry_no]
    else:
        return None

# Function to compute persistent binnng features
def compute_binning(dim_n, thresholds):
    n = np.size(dim_n, 0)
    intersects = []
    
    if(n > 0):
        if(np.size(thresholds) == 1):
            thresholds = [thresholds]
            
        for threshold in thresholds:
            int_count = 0
            for i in range(n):
                if(threshold >= dim_n[i,0] and threshold <= dim_n[i,1]):
                    int_count += 1
                    
            intersects.append(int_count)
        
    return intersects

# Function to compute persistent statistics features
def get_barcode_stats(barcode):
    # Computing Statistics from Persistent Barcodes

    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        bc_std0, bc_std1 = np.std(barcode, axis=0)
        # Median of Birth and Death of the barcode
        bc_med0, bc_med1 = np.median(barcode, axis=0)
        diff_barcode = np.subtract([i[1] for i in barcode], [i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars        
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        bc_lengthSTD = np.std(diff_barcode)
        # Median of length of Bars
        bc_lengthMedian = np.median(diff_barcode)
        # Number of Bars
        bc_count = len(diff_barcode)

        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                     bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_count])
    else:
        bar_stats= np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    bar_stats[~np.isfinite(bar_stats)] = 0
    
    return bar_stats

# Function to compute persistent image features
def get_pers_imgs(barcode, persIm):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = persIm.fit_transform([barcode])
    
    return feature_vectors

# Function to compute persistent landscape features
def get_pers_lands(barcode, persLand):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = persLand.fit_transform([barcode])
    
    return feature_vectors

# Function to compute persistent diagrams
def get_pds(img, lbp_idx, max_dim):
    feature_vectors = []
    dist_mat = ulbp_top_basis(img, lbp_idx)
    
    if(np.size(dist_mat) != 0):
        ripser_result = ripser(dist_mat, maxdim=max_dim, thresh=50, distance_matrix=True)
        feature_vectors = ripser_result['dgms']
    
    return feature_vectors

# Function to compute featurized barcodes
def get_pd_featurized_barcodes(imgPath):
    print(imgPath)
    
    dim0_bin = [[[] for i in range(8)] for j in range(7)]
    dim0_stats = [[[] for i in range(8)] for j in range(7)]
    dim1_bin = [[[] for i in range(8)] for j in range(7)]
    dim1_stats = [[[] for i in range(8)] for j in range(7)]
    dim0_persims = [[[] for i in range(8)] for j in range(7)]
    dim1_persims = [[[] for i in range(8)] for j in range(7)]
    dim0_perland = [[[] for i in range(8)] for j in range(7)]
    dim1_perland = [[[] for i in range(8)] for j in range(7)]

    # Binning range
    binRange = np.arange(0, 30)

    # Persistent Image
    persIm = representations.PersistenceImage(resolution=[30, 30])

    # Persistent Landscape
    persLand = representations.Landscape(resolution=100)
    
    img = cv2.imread(imgPath,0)

    for geo in range(7):
        i_lbp = get_geomtetry_ulbp(geo)
        
        for rot in range(8):
            feature_vectors = get_pds(img, i_lbp[rot], 1)

            fig, axs = plt.subplots(1, 3, figsize=(10,5))

            axs[0].set_title("Original Diagram")
            pimgr.plot_diagram(dim0[0], skew=False, ax=axs[0])
            
            if(np.size(feature_vectors) == 0):
                continue
            
            dim0 = feature_vectors[0]
            dim0 = dim0[dim0[:, 0] != np.inf]
            dim0 = dim0[dim0[:, 1] != np.inf]

            dim1 = feature_vectors[1]
            dim1 = dim1[dim1[:, 0] != np.inf]
            dim1 = dim1[dim1[:, 1] != np.inf]
            
            dim0_bin[geo][rot] = compute_binning(dim0, binRange)
            dim0_stats[geo][rot] = get_barcode_stats(dim0)
            
            dim1_bin[geo][rot] = compute_binning(dim1, binRange)
            dim1_stats[geo][rot] = get_barcode_stats(dim1)
            
            dim0_persims[geo][rot] = get_pers_imgs(dim0, persIm)
            dim1_persims[geo][rot] = get_pers_imgs(dim1, persIm)
            
            dim0_perland[geo][rot] = get_pers_lands(dim0, persLand)
            dim1_perland[geo][rot] = get_pers_lands(dim1, persLand)
    
    return [os.path.basename(imgPath), dim0_bin, dim0_stats, dim1_bin, dim1_stats, dim0_persims, dim1_persims, dim0_perland, dim1_perland]


# Main thread to run the script
if __name__ == '__main__':
    root_directory = '/Users/pertsemlidish22/tda/images'
    file_extension = '*.gif'
    file_pattern = os.path.join(root_directory, file_extension)
    outputFolder = 'ExportedFeatures'
    for root, dirs, files in os.walk(root_directory):
        if root != root_directory:
            for filename in files:
                image_path = os.path.join(root, filename)
                result = get_pd_featurized_barcodes(image_path)
                np.save(f'{outputFolder}//', result)


    X = np.load("ExportedFeatures/DDSM_Mass_257images/CSV/G0/R0/dim0_bin.npy" )
    print(X)
    Y = np.load("ExportedFeatures/DDSM_Mass_257images/CSV/G0/R1/dim0_bin.npy" )
    distanceXY = gudhi.wasserstein.wasserstein_distance(X, Y, matching=False, order=1.0, internal_p= 2.0, enable_autodiff=False, keep_essential_parts=True)
    print(distanceXY)