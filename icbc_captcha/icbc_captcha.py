import csv
import numpy as np
from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

path = "Pics_5000"
csvfile = open(path+'/labels.csv')
reader = csv.reader(csvfile)

labels = []
for line in reader:
    tmp = [line[0], line[1]]
    labels.append(tmp)
csvfile.close()
# print labels

X = []
Y = []
picnum = len(labels)


for i in range(picnum):
    # img = cv2.imread(path+'/' + labels[i][0] + '.png', cv2.IMREAD_GRAYSCALE)
    img = Image.open(path+'/' + labels[i][0] + '.png').convert('L')
    img = np.array(img)
    X.append(img)
    Y.append(labels[i][1])

X = np.array(X)

labeldict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16,
             'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23,
             'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29,
             'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}

num_classes =36
X = np.array(X)
for i in range(len(Y)):
    c0 = keras.utils.to_categorical(labeldict[Y[i][0]], num_classes)
    c1 = keras.utils.to_categorical(labeldict[Y[i][1]], num_classes)
    c2 = keras.utils.to_categorical(labeldict[Y[i][2]], num_classes)
    c3 = keras.utils.to_categorical(labeldict[Y[i][3]], num_classes)
    c = np.concatenate((c0, c1, c2, c3), axis=1)
    Y[i] = c

Y = np.array(Y)
Y = Y[:,0,:]

batch_size = 128
epochs = 50
img_rows, img_cols = 36, 90
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
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

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
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(num_classes*4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save('Model/icbc_5000.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred = model.predict(x_test,batch_size = 32)
outdict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
correct_num = 0
for i in range(pred.shape[0]):
    c0 = outdict[np.argmax(pred[i][:36])]
    c1 = outdict[np.argmax(pred[i][36:36*2])]
    c2 = outdict[np.argmax(pred[i][36*2:36*3])]
    c3 = outdict[np.argmax(pred[i][36*3:])]
    c = c0+c1+c2+c3
    if c == labels[4600+i][1]:
        correct_num = correct_num + 1
print("Test Whole Accurate : ", float(correct_num)/len(pred))
