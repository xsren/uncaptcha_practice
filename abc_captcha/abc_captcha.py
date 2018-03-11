import csv
import keras
import numpy as np
from PIL import Image
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten


path = "Pics"
with open('Model/labels_13k.csv') as csvfile:
    reader = csv.reader(csvfile)
    labels = []
    for line in reader:
        tmp = [line[0], line[1]]
        labels.append(tmp)

# 把图片和对应打标签分别放入两个列表
X = []
Y = []
pic_num = len(labels)
for i in range(pic_num):
    img = Image.open(path+'/' + eval(labels[i][0]).decode('utf-8') + '.jpg').convert('L')
    img = np.array(img)
    X.append(img)
    Y.append(eval(labels[i][1]).decode('utf-8'))
X = np.array(X)

# 将26个分类转换成one-hot编码
label_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
              'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
              'y': 24, 'z': 25}
num_classes = 26
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

# 训练所用参数
batch_size = 32
epochs = 150
img_rows, img_cols = 30, 100
x_train = X[:12000]
y_train = Y[:12000]
x_test = X[12000:]
y_test = Y[12000:]

# 检查后端, 设置所需的shape
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)

# 归一化处理
x_train = 255 - x_train
x_test = 255 - x_test
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Create a model
model = Sequential()
# Layer1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Full connect Layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes*4, activation='softmax'))
# Compile a model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Display the model
model.summary()
# Train and save the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.save('Model/abc_13k.h5')
