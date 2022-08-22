import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)



#model = tf.keras.models.load_model('mnist.h5')
model = tf.keras.models.load_model('huan_luyen.model')

image_number = 1
while os.path.isfile(f"anh_thu_nghiem/img{image_number}.png"):
    try:
        img = cv2.imread(f"anh_thu_nghiem/img{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Đó là số: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Lỗi!")
    finally:
        image_number += 1

