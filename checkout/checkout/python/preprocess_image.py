import cv2
import keras.backend as K
import keras.preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input

#funkcja dokonująca wstępnego przetworzenia zdjęcia
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image