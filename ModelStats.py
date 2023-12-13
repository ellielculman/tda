import numpy as np
from sklearn.metrics import precision_score, accuracy_score



# Assuming a 3x3 confusion matrix
conf_matrix = np.array([
    [56, 58, 42],
    [71, 82, 40],
    [37, 36, 35]
])

accuracy = (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2])/(np.sum(conf_matrix))
print(accuracy)

