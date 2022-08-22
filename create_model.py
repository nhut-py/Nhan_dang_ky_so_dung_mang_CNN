# import thư vien
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from tensorflow import keras

number = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = number.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

for train in range(len(x_train)):
    for row in range(28):
        # kick thuoc anh
        for x in range(28):
            if x_train[train][row][x] != 0:
                x_train[train][row][x] = 1

model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

# Chuyen mang 28x28 thanh mang 1 hang va 28x28 cot
# ham kich hoat relu chi nhan gia tri dau vao > 0, tra ve 0 voi cac so con lai
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))

#lop cuoi cung dung ham softmax co kich thuoc la 10
#ham softmax co tac dung chon ra gia tri lon nhat trong tap so
model.add(tf.keras.layers.Dense(10, activation= tf.nn.softmax))

#xay dung mang no-ron
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=200)
model.save('huan_luyen_2.model')
print("Đã lưu huấn luyện")