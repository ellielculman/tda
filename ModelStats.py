import numpy as np
from sklearn.metrics import precision_score, accuracy_score
import Model



# Assuming a 3x3 confusion matrix
conf_matrix = np.array([
    [56, 58, 42],
    [71, 82, 40],
    [37, 36, 35]
])

accuracy = (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2])/(np.sum(conf_matrix))
print(accuracy)

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Assuming you have test_labels (y_test) and predicted_labels (y_pred)
y_test = test_labels  # Replace with your actual labels
y_pred = Mclassify_arr  # Replace with your model's predictions

# Assuming you have class labels
classes = ["benign", "cancer", "normal"]

# Encode class labels
class_encoding = {class_label: i for i, class_label in enumerate(classes)}

# Convert labels to numeric values
y_test_numeric = [class_encoding[label] for label in y_test]
y_pred_numeric = [class_encoding[label] for label in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test_numeric, y_pred_numeric)

# Calculate precision for each class
precision_per_class = precision_score(y_test_numeric, y_pred_numeric, average=None)

# Calculate overall precision (macro-average)
overall_precision = precision_score(y_test_numeric, y_pred_numeric, average='macro')

# Create confusion matrix
conf_matrix = confusion_matrix(y_test_numeric, y_pred_numeric)

print("Accuracy:", accuracy)
print("Precision per class:", precision_per_class)
print("Overall Precision (Macro-average):", overall_precision)
print("Confusion Matrix:")
print(conf_matrix)


