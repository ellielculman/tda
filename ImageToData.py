from PIL import Image
import matplotlib.pyplot as plt

image_path = "tda/images/Volume: cancer_08 Case: A-1503-1/A_1503_1.LEFT_MLO.LJPEG.1_highpass.gif"

# Open the image
img = Image.open(image_path)

# Display the image
plt.imshow(img, cmap="gray")  # Specify cmap="gray" for grayscale images
plt.title("Black and White Image")
plt.show()