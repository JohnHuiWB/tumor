from keras.models import  Model,load_model
from keras.layers import  GlobalAveragePooling2D, Dense
from keras.callbacks import ModelCheckpoint, History
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD,Adam
from keras.utils.vis_utils import plot_model
import cv2
import numpy as np
from gen import FILENAME_V,FILENAME,FILENAME_T,generate_arrays_from_file,NUM_TEST
from keras import backend as K
class Inception(object):
    def __init__(self):
        self.model_path = 'model.h5'
        
    def draw_model(self,model):
        plot_model(model, to_file='model.jpg', show_shapes=True)
        print(self.model_path)

    def inception(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x) #new FC layer, random init
        predictions = Dense(1, activation='sigmoid')(x) #new softmax layer
        for layer in base_model.layers:
            layer.trainable = False
        model = Model(input=base_model.input, output=predictions)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self,model, batch_size=8, samples_per_epoch=500, epochs=16):  #8,2000,5
        # history = History()
        model_checkpoint = ModelCheckpoint(
                self.model_path, monitor='val_loss',verbose=1, save_best_only=True)
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
        # model.save('vgg16.h5' + '.' + str(history.history['val_loss'][-1]))
        print('save to '+ self.model_path)

    def continue_train(self,filename,batch_size=4, samples_per_epoch=200, epochs=3):
        model=load_model(filename)
        model_checkpoint = ModelCheckpoint(
                self.model_path, monitor='val_loss',verbose=1, save_best_only=True)
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
        print('save to '+ self.model_path)


    def test_one(self,filename):
        model =load_model(filename)
        x, y = generate_arrays_from_file(FILENAME_T,1).__next__()
        r =  model.predict(x)
        x *= 255
        x = x.astype('uint8')
        x = np.squeeze(x)
        print(y,r)
        im = cv2.imshow('x', x)
        cv2.waitKey(0)

    def test_three(self,filename):
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

    def eval(self,filename):
        model =load_model(filename)
        g = generate_arrays_from_file(FILENAME_T)
        result = model.evaluate_generator(g, steps=200)
        return result

    def test_10(self,filename):
        model =load_model(filename)
        for x,y in generate_arrays_from_file(FILENAME_T,24):
            r = model.predict(x)
            print(y,r)
            break

if __name__ == '__main__':
    # M=Inception()
    # M.draw_model(load_model('model.h5'))
