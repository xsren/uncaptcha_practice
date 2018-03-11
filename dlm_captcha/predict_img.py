from PIL import Image
import numpy as np
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten


class Predict(object):

    def __init__(self):
        self.model = self.create_model()
        model_path = 'Model/dlm_4600.h5'
        self.model.load_weights(model_path)

    @staticmethod
    def initialize_img(img_path):
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        X = [img]
        X = np.array(X)
        img_rows, img_cols = 34, 80
        x_train = X

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

        x_train = 255 - x_train
        x_train = x_train.astype('float32')
        x_train /= 255
        return x_train

    @staticmethod
    def create_model():

        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(34, 80, 1)))
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
        model.add(Dense(10 * 4, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def predict_img(self, img_path):
        tmp = self.initialize_img(img_path)
        pred = self.model.predict(tmp, batch_size=32)
        out_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        c0 = out_dict[np.argmax(pred[0][:10])]
        c1 = out_dict[np.argmax(pred[0][10:10 * 2])]
        c2 = out_dict[np.argmax(pred[0][10 * 2:10 * 3])]
        c3 = out_dict[np.argmax(pred[0][10 * 3:])]
        c = c0 + c1 + c2 + c3
        return c


if __name__ == '__main__':
    bar = Predict()
    bar.predict_img('0.png')
