from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
import os
import gudhi.wasserstein
import scipy.stats

def test_one(tester):
    
    minVal_benign = 1000000000000
    minIndex_benign = 0
    indx_benign = 0
    minVal_cancer = 1000000000000
    minIndex_cancer = 0
    indx_cancer = 0
    minVal_normal = 1000000000000
    minIndex_normal = 0
    indx_normal = 0

    train_data_benign = []
    train_data_cancer = []
    train_data_normal = []
    i = 0
    for data in train_data:
        if train_labels[i] == "benign":
            train_data_benign.append(data)
        if train_labels[i] == "cancer":
            train_data_cancer.append(data)
        if train_labels[i] == "normal":
            train_data_normal.append(data)
    for val_benign in train_data_benign :
        distanceXY = scipy.stats.wasserstein_distance(tester, val_benign, u_weights=None, v_weights=None)    
        if distanceXY < minVal_benign:
            minVal_benign = distanceXY
            minIndex_benign = indx_benign
        indx_benign += 1
    for val_cancer in train_data_cancer :
        distanceXY = scipy.stats.wasserstein_distance(tester, val_cancer, u_weights=None, v_weights=None)    
        if distanceXY < minVal_cancer:
            minVal_cancer = distanceXY
            minIndex_cancer = indx_cancer
        indx_cancer += 1
    for val_normal in train_data_normal :
        distanceXY = scipy.stats.wasserstein_distance(tester, val_normal, u_weights=None, v_weights=None)    
        if distanceXY < minVal_normal:
            minVal_normal = distanceXY
            minIndex_normal = indx_normal
        indx_normal+= 1
    total_min = min(minVal_benign, minVal_cancer, minVal_normal)
    if total_min == minVal_benign: 
        class_pred = "benign"
    if total_min == minVal_cancer: 
        class_pred = "cancer"
    if total_min == minVal_normal: 
        class_pred = "normal"
    return class_pred

def test_mod():

    classify_arr = []
    for test in test_data:
        classify_arr.append(test_one(test))
        print(test_one(test))

    benign_list = [0, 0, 0]
    cancer_list = [0, 0, 0]
    normal_list = [0, 0, 0]
    indx = 0
    for item in classify_arr:
        if item == "benign" and test_labels[indx] == "benign":
            benign_list[0] += 1
        if item == "benign" and test_labels[indx] == "cancer":
            benign_list[1] += 1
        if item == "benign" and test_labels[indx] == "normal":
            benign_list[2] += 1
        if item == "cancer" and test_labels[indx] == "benign":
            cancer_list[0] += 1
        if item == "cancer" and test_labels[indx] == "cancer":
            cancer_list[1] += 1
        if item == "cancer" and test_labels[indx] == "normal":
            cancer_list[2] += 1
        if item == "normal" and test_labels[indx] == "benign":
            normal_list[0] += 1
        if item == "normal" and test_labels[indx] == "cancer":
            normal_list[1] += 1
        if item == "normal" and test_labels[indx] == "normal":
            normal_list[2] += 1
        indx +=1

    print(benign_list)
    print(cancer_list)
    print(normal_list)


if __name__ == '__main__':
    folder = "ExportedFeatures"
    np.random.seed(2)  # Setting seed for reproducibility
    data_list = []
    label_list = []

    # Traverse subdirectories
    min_size = 100000000000000000000000
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)

                # Load numpy file
                features = np.load(file_path)
                if min(features.shape) < min_size:
                    min_size = min(features.shape)
                features = np.resize(features, min_size)

                # Extract label from folder name (adjust this based on your folder structure)
                parent_folder = os.path.basename(os.path.dirname(root))
                label = parent_folder

                data_list.append(features)
                label_list.append(label)
    # Ensure 'class' is a categorical variable for stratified sampling
    data_array = np.array(data_list)
    label_array = np.array(label_list, dtype='str')  # Convert labels to string type
    label_array = label_array.reshape(-1, 1)  # Reshape to a column vector
    label_array = pd.Categorical(label_array.squeeze())

    test_size = 0.25
    if test_size * len(np.unique(label_array)) >= len(label_array):
        raise ValueError("Invalid test_size. It should be less than 1 and should result in a non-empty test set.")

    # Stratified train-test split with 75% training and 25% testing
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_array, label_array, test_size=0.25, random_state=2, stratify=label_array
    ) 
    # Display class distribution in the training and testing sets
    #print("Training set distribution:")
    #print(pd.Series(train_labels).value_counts(normalize=True))

    #print("\nTesting set distribution:")
    #print(pd.Series(test_labels).value_counts(normalize=True))
    #test_one(test_data[43])
    test_mod()
