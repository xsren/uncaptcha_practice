import csv
import numpy as np
from PIL import Image

import keras
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten


path = "Pics"
with open('Model/labels.csv') as csvfile:
    reader = csv.reader(csvfile)
    labels = []
    for line in reader:
        tmp = [line[0], line[1]]
        labels.append(tmp)

X = []
Y = []
pic_num = len(labels)
for i in range(pic_num):
    img = Image.open(path+'/' + eval(labels[i][0]).decode('utf-8') + '.png')
    img = np.array(img)
    X.append(img)
    Y.append(eval(labels[i][1]).decode('utf-8'))
X = np.array(X)

label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
num_classes = 10
X = np.array(X)
for i in range(len(Y)):
    c0 = keras.utils.to_categorical(label_dict[Y[i][0]], num_classes)
    c1 = keras.utils.to_categorical(label_dict[Y[i][1]], num_classes)
    c2 = keras.utils.to_categorical(label_dict[Y[i][2]], num_classes)
    c3 = keras.utils.to_categorical(label_dict[Y[i][3]], num_classes)
    c = np.concatenate((c0, c1, c2, c3), axis=1)
    Y[i] = c
Y = np.array(Y)
Y = Y[:, 0, :]

batch_size = 32
epochs = 150
img_rows, img_cols = 34, 80
x_train = X[:4600]
y_train = Y[:4600]
x_test = X[4600:]
y_test = Y[4600:]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = 255 - x_train
x_test = 255 - x_test
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes*4, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save('Model/dlm_4600.h5')
