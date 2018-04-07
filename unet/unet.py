#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : Unet.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  :
# @Contact : huiwenbin199822@gmail.com
# @Software : PyCharm

from os import path
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, History
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from tumor import data


class Unet(object):
    def __init__(self):
        self._dir = path.dirname(path.realpath(__file__))
        self.model_path = path.join(self._dir, 'unet.h5')
        self.smooth = 1.

    def draw_model(self):
        model = self._get_model()
        from keras.utils.vis_utils import plot_model
        # 神经网络可视化
        filename = path.join(
            path.dirname(
                path.realpath(__file__)),
            'model.png')
        plot_model(model, to_file=filename, show_shapes=True)

    @staticmethod
    def _get_model():
        inputs = Input(data.PIXELS)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
            2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
            2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
            2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.compile(
            optimizer=Adam(
                lr=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model

    def train(self, batch_size=1, samples_per_epoch=1000, epochs=1):
        model = self._get_model()
        # model_checkpoint = ModelCheckpoint(
        #     self.model_path, monitor='val_loss', save_best_only=True)
        history = History()
        model.fit_generator(
            data.generate_arrays_from_file(data.FILENAME, batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=data.generate_arrays_from_file(data.FILENAME_V),
            validation_steps=20,
            shuffle=True,
            callbacks=[#model_checkpoint,
                       history]
        )
        model.save(self.model_path + '.' + str(history.history['val_loss'][-1]))
        print('save to '+ 'unet.h5'+'.'+str(history.history['val_loss'][-1]))

    def continue_train(self, model_path, batch_size=1, samples_per_epoch=1000, epochs=1):
        model = load_model(path.join(self._dir, model_path))
        # model_checkpoint = ModelCheckpoint(
        #     self.model_path, monitor='val_loss', save_best_only=True)
        history = History()
        model.fit_generator(
            data.generate_arrays_from_file(data.FILENAME, batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=data.generate_arrays_from_file(data.FILENAME_V),
            validation_steps=20,
            shuffle=True,
            callbacks=[#model_checkpoint,
                       history]
        )
        model.save(self.model_path+'.'+str(history.history['val_loss'][-1]))
        print('save to '+ 'unet.h5'+'.'+str(history.history['val_loss'][-1]))

    def _load_model(self):
        return load_model(self.model_path)

    def predict(self, x):
        model = self._load_model()
        result = model.predict(x, verbose=1)
        return result

    def eval(self):
        model = self._load_model()
        g = data.generate_arrays_from_file(data.FILENAME_T)
        result = model.evaluate_generator(g, steps=data.NUM_TEST)
        return result


if __name__ == '__main__':
    u = Unet()
    # u.draw_model()
    # u.train(batch_size=1, samples_per_epoch=1, epochs=1)
    # u.continue_train('unet.h5.0.6830260127782821', batch_size=1, samples_per_epoch=1, epochs=1)