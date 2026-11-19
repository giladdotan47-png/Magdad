import cv2
import numpy as np

def visualData(data, label):
    # Create a 350x700 black image (height, width, channels)
    img = np.zeros((1920, 1080, 3), dtype=np.uint8)

    # Display the image
    cv2.imshow('Black Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

