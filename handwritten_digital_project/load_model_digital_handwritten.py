import numpy as np
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

# Prediction

def resize_image(image):
    image_array = cv2.imread(image)[:, :, 0]
    image_resize = np.invert(np.array([image_array]))
    predict_image = model.predict(image_resize)

    print(f"This digit is probably a {np.argmax(predict_image)}")

    np_array = np.array(image_resize)
    plt.imshow(np_array[0], cmap=plt.cm.binary)
    plt.show()

# testing

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

model = keras.models.load_model("handwritten_digit_model.h5")
resize_image('num_0.png')
resize_image('num_1.png')
resize_image('num_2.png')
resize_image('num_3.png')
resize_image('num_5.png')
resize_image('num_7.png')
resize_image('num_8.png')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
