import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

class Predict(object):

    def __init__(self):
        self.model = self.create_model()
        model_path = 'Model/hurong_captcha.h5'
        self.model.load_weights(model_path)

    def initialize_img(self, img_path):
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        X = [img]
        X = np.array(X)
        img_rows, img_cols = 50, 130
        x_train = X

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

        x_train = 255 - x_train
        x_train = x_train.astype('float32')
        x_train /= 255
        num_classes = 36
        return x_train


    def create_model(self):

        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 130, 1)))
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
        model.add(Dense(36 * 4, activation='softmax'))
        return model


    def predict_img(self, img_path):
        tmp = self.initialize_img(img_path)
        pred = self.model.predict(tmp, batch_size=32)
        outdict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

        c0 = outdict[np.argmax(pred[0][:36])]
        c1 = outdict[np.argmax(pred[0][36:36 * 2])]
        c2 = outdict[np.argmax(pred[0][36 * 2:36 * 3])]
        c3 = outdict[np.argmax(pred[0][36 * 3:])]
        c = c0 + c1 + c2 + c3
        print(c)
        return c

if __name__ == '__main__':
    bar = Predict()
    bar.predict_img('code1.jpg')
    bar.predict_img('code2.jpg')
    bar.predict_img('code3.jpg')
    bar.predict_img('code4.jpg')
