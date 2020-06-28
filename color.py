# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:10:37 2020

@author: Pablo
"""

from keras.layers import Convolution2D, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from keras.models import load_model
from model import Model
from skimage.io import imsave
import numpy as np
import os
import random
from skimage import measure 

import time
from PIL import Image

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


tf.python.control_flow_ops = tf

# Image transformer
datagen = ImageDataGenerator(
		rescale=1.0/255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

# Get images
X = []
dim=(128,128)
for folder in os.listdir('tiny2/train'):
    print(folder)
    if len(os.listdir('tiny2/train/' + folder) ) != 0 :
        for filename in os.listdir('tiny2/train/'+folder+'/images/'):
            try:
                X.append(img_to_array(load_img('tiny2/train/'+folder+ '/images/' + filename)))
            except:
                print('cant identify')


X = np.array(X)

np.random.shuffle(X)


# Set up train and test data

XtestOriginal= []

split = int(0.9*len(X))
Xtrain = X[:split]

#XtestOriginal = X[split:]
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))

Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]

print('preparing')

#Train = rgb2lab(1.0/255*X[0:23328])[:,:,:,0]
#Train = Train.reshape(Train.shape+(1,))
#TrainY = rgb2lab(1.0/255*X[0:23328])[:,:,:,1:]



print('prepared')
start_time = time.time()



# Generate training data
batch_size = 128
def image_a_b_gen(batch_size):
	for batch in datagen.flow(Xtrain, batch_size=batch_size):
		if batch.any() == None:
			break		
		lab_batch = rgb2lab(batch)
		X_batch = lab_batch[:,:,:,0]
		Y_batch = lab_batch[:,:,:,1:] / 128
		yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model

      
model = Model()



history = model.fit_generator(
	image_a_b_gen(batch_size),
	epochs=1,
	steps_per_epoch=1)

# Test model

print('HISTORY')

plt.plot(history.history['accuracy'])
plt.title('model accuracy')

plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

fig1 = plt.gcf()

plt.show()
plt.draw()
fig1.savefig('acc2.png', dpi=100)
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig2 = plt.gcf()

plt.show()
plt.draw()
fig2.savefig('loss2.png', dpi=100)




#print(model.evaluate(Xtest, Ytest))
output = model.predict(Xtest)
output = output * 128
# Output colorizations

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (64, 128))
    dst.paste(im1)
    dst.paste(im2)
    return dst

def mse_img(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse_img(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB, multichannel=True)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB)
	plt.axis("off")
	# show the images
	plt.show()



for i in range(len(output)):
	cur = np.zeros((64, 64, 3))
	ori = np.zeros((64, 64, 3))
	ori[:,:,0] = Xtest[i][:,:,0]
	ori[:,:,1:] = Ytest[i][:,:,1:]
	cur[:,:,0] = Xtest[i][:,:,0]
	cur[:,:,1:] = output[i]
	original = lab2rgb(ori)
	colorized_output = lab2rgb(cur)
	compare_images(original, colorized_output, "Original vs. Contrast")
	imsave("color/img_"+str(i)+".png", lab2rgb(cur))
	imsave("color/img_gray_"+str(i)+".png", rgb2gray(lab2rgb(cur)))





print("--- %s seconds ---" % (time.time() - start_time))
