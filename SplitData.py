from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
import os

 

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
                    min_size = features.shape
                features = np.resize(features, min_size)

                # Extract label from folder name (adjust this based on your folder structure)
                label = os.path.basename(root)

                data_list.append(features)
                label_list.append(label)

    # Ensure 'class' is a categorical variable for stratified sampling
    data_array = np.array(data_list)
    label_array = np.array(label_list, dtype='str')  # Convert labels to string type
    label_array= pd.Categorical(label_array)

    # Stratified train-test split with 75% training and 25% testing
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_array, label_array, test_size=0.25, random_state=2, stratify=label_array
    ) 
    # Display class distribution in the training and testing sets
    print("Training set distribution:")
    print(pd.Series(train_labels).value_counts(normalize=True))

    print("\nTesting set distribution:")
    print(pd.Series(test_labels).value_counts(normalize=True))