from keras.models import  Model,load_model
from keras.layers import Flatten, Dense, Dropout,Input,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import SGD,Adam
from keras.utils.vis_utils import plot_model
import cv2
import numpy as np
from gen import FILENAME_V,FILENAME,FILENAME_T,generate_arrays_from_file,NUM_TEST
from keras import backend as K

def draw_model(model):
	plot_model(model, to_file='model.jpg', show_shapes=True)

def VGG_16():
    img_input = Input(shape=(512,512,1))
    # Block 1
    x = Conv2D(64, (66, 66),strides=(2,2), activation='relu', padding='valid', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(img_input, x, name='vgg16')
    
    # adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = SGD(decay=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
    return model

def train(model, batch_size=2, samples_per_epoch=100, epochs=3):  #8,2000,5
    # history = History()
    model_checkpoint = ModelCheckpoint(
            'model.h5', monitor='val_loss',verbose=1, save_best_only=True)
    model.fit_generator(
        generate_arrays_from_file(FILENAME, batch_size),
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        verbose=1,
        validation_data=generate_arrays_from_file(FILENAME_V),
        validation_steps=50,
        shuffle=True,
        callbacks=[model_checkpoint]
    )
    # model.save('vgg16.h5' + '.' + str(history.history['val_loss'][-1]))
    print('save to '+ 'model.h5')

def continue_train(filename,batch_size=2, samples_per_epoch=100, epochs=3):
    model=load_model(filename)
    model_checkpoint = ModelCheckpoint(
            'model.h5', monitor='val_loss',verbose=1, save_best_only=True)
    model.fit_generator(
        generate_arrays_from_file(FILENAME, batch_size),
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        verbose=1,
        validation_data=generate_arrays_from_file(FILENAME_V),
        validation_steps=100,
        shuffle=True,
        callbacks=[model_checkpoint]
    )
    print('save to '+ 'model.h5')


def test_one(filename):
    model =load_model(filename)
    x, y = generate_arrays_from_file(FILENAME_T,1).__next__()
    r =  model.predict(x)
    x *= 255
    x = x.astype('uint8')
    x = np.squeeze(x)
    print(y,r)
    im = cv2.imshow('x', x)
    cv2.waitKey(0)

def test_three(filename):
    model =load_model(filename)
    i=1
    for x,y in generate_arrays_from_file(FILENAME_T,1):
        r = model.predict(x)
        print(y,r)
        x *= 255
        # 还原为uint8类型
        x = x.astype('uint8')
        # 删掉维数为1的维度，保留(512, 512)的矩阵
        x = np.squeeze(x)
        cv2.imshow('x'+str(i), x)
        i+=1
        if(i>3):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def eval(filename):
    model =load_model(filename)
    g = generate_arrays_from_file(FILENAME_T)
    result = model.evaluate_generator(g, steps=200)
    return result

def test_10(filename):
    model =load_model(filename)
    for x,y in generate_arrays_from_file(FILENAME_T,20):
        r = model.predict(x)
        print(y,r)
        break

if __name__ == '__main__':
	# print(test_10('model.h5'))
	# print(eval('model.h5'))
	# test_three('model.h5')
	train(VGG_16())
	# continue_train('model.h5')