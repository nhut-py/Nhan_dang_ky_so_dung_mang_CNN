#import thu vien
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = models.Sequential()
# Chuyen mang 28x28 thanh mang 1 hang va 28x28 cot
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#lop cuoi cung dung ham softmax co kich thuoc la 10
#ham softmax co tac dung chon ra gia tri lon nhat trong tap so
model.add(layers.Dense(10, activation='softmax'))
model.summary()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Dự đoán tỉ lệ chính xác: ", test_acc)
print("Dự đoán tỉ lệ thất bại: ", test_loss)
x = np.array([" Dự đoán tỉ lệ thành công", "Dự đoán tỉ lệ thất bại"])
y = np.array([test_acc, test_loss])

plt.bar(x,y)
plt.show()

model.save('training/mnistss.h5') # Training phương pháp học rmsprop
#model.save('training/adam.h5') # Training phương pháp học adam
