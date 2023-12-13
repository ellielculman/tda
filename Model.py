from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
import os
import gudhi.wasserstein
import scipy.stats

def test_one(tester):
    
    minVal = 1000000000000
    minIndex = 0
    indx = 0

    for val in train_data:
        distanceXY = scipy.stats.wasserstein_distance(tester, val, u_weights=None, v_weights=None)    
        if distanceXY < minVal:
            minVal = distanceXY
            minIndex = indx
        indx += 1
    #print(minVal)
    #print(minIndex)
    class_pred = train_labels[minIndex]
    #print("class predicted: " + class_pred)
    #print("true class: " + test_labels[310])
    return class_pred

def test_mod():

    classify_arr = []
    for test in test_data:
        classify_arr.append(test_one(test))

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
    
    test_mod()
