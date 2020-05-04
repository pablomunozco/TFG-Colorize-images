

# imports
from skimage.color import rgb2gray,lab2rgb, rgb2lab
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from skimage.io import imsave
import numpy as np
import os




# -----------Variables-------------

data = []
labels = []
data_bw= []
image_lab = []
colorize = []
count = 0
path = 'C:/Users/Pablo/Desktop/tfg-colorize/train/'
prueba = 'C:/Users/Pablo/Desktop/tfg-colorize/prueba/'
pathBw = 'C:/Users/Pablo/Desktop/tfg-colorize/bw/'


#rgb_weights = [0.2989, 0.5870, 0.1140]

print("loading images...")



img=[]


# loading images for train
for filename in os.listdir(path):
    img.append(img_to_array(load_img(path+filename)))
    
img = np.array(img, dtype=float)

split = int(0.95*len(img))
imgTrain = img[:split]
imgTrain = 1.0/255*imgTrain

#dim = (width, height)
#
#resized_img = cv2.resize(image, dim)
#
#h, w, c = resized_img.shape
#
#print(resized_img.shape)


    
print("images loaded")


#create a model 
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')



# Image transformer
dataGen = ImageDataGenerator(shear_range=0.4, zoom_range=0.4, rotation_range=10, horizontal_flip=True)

# Generate training data
batch_size = 12
def set_chanels_a_b(batch_size):
    for batch in dataGen.flow(imgTrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_bw = lab_batch[:,:,:,0]
        Y_ab = lab_batch[:,:,:,1:] / 128
        yield (X_bw.reshape(X_bw.shape+(1,)), Y_ab)

# Train model      
model.fit_generator(set_chanels_a_b(batch_size), epochs=12, steps_per_epoch=12)




# Test images
imgTestBw = rgb2lab(1.0/255*img[split:])[:,:,:,0] #bw
imgTestBw = imgTestBw.reshape(imgTestBw.shape+(1,))
imgTestColor = rgb2lab(1.0/255*img[split:])[:,:,:,1:] #color
imgTestColor = imgTestColor / 128


for filename in os.listdir(pathBw):
    colorize.append(img_to_array(load_img(pathBw+filename)))
    
    
colorize = np.array(colorize, dtype=float)
colorize = rgb2lab(1.0/255*colorize)[:,:,:,0]
colorize = colorize.reshape(colorize.shape+(1,))



print('Testing...')

output = model.predict(colorize)
output = output * 128

# Output colorizations
for i in range(len(output)):
    img_colorized = np.zeros((256, 256, 3))
    img_colorized[:,:,0] = colorize[i][:,:,0]
    img_colorized[:,:,1:] = output[i].astype(np.uint8)
    imsave("output/img_colorized"+str(i)+".png", lab2rgb(img_colorized))
    
    

