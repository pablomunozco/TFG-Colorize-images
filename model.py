from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from skimage.io import imsave
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from tensorflow.python.keras import models
from keras.utils.vis_utils import plot_model


from keras.models import Sequential



class Model( object ):

    def __init__(self):


        N = 5
        model = Sequential()
        num_maps1 = [4, 8, 16, 32, 64]
        num_maps2 = [8, 16, 32, 64, 128]
        for i in range(N):
            if i == 0:
                model.add(Conv2D(num_maps1[i], 3, 3, border_mode='same', subsample=(2, 2), input_shape=(64, 64, 1)))
            else:
                model.add(Conv2D(num_maps1[i], 3, 3, border_mode='same', subsample=(2, 2)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Conv2D(num_maps2[i], 3, 3, border_mode='same', subsample=(1, 1)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))


        for i in range(N):
            model.add(UpSampling2D(size=(2, 2)))
            model.add(Conv2D(num_maps2[-(i+1)], 3, 3, border_mode='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            if i != N-1:
                model.add(Conv2D(num_maps1[-(i+1)], 3, 3, border_mode='same'))
                model.add(BatchNormalization())
                model.add(Activation('tanh'))
            else:
                model.add(Conv2D(2, 3, 3, border_mode='same'))

        model.compile(optimizer='rmsprop',
            loss='mse',metrics=['accuracy'])
        
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
        self.__model = model
        

    def fit(self, X, Y, number_of_epochs):
        self.__model.fit(X, Y, batch_size=3 , epochs=number_of_epochs)
        self.__model.summary()

    def evaluate(self, test_X, test_Y):
        return self.__model.evaluate(test_X, test_Y,batch_size=10)

    def predict(self, X):
        predictions = self.__model.predict(X)
        return predictions

    def save_model(self, file_path):
        self.__model.save(file_path)
        
    def fit_generator(self, images, epochs, steps_per_epoch):
        history = self.__model.fit_generator(images,epochs,steps_per_epoch)
        return history

    def load_model(self, file_path):
        self.__model = models.load_model(file_path)