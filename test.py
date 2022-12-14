# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import os
import cv2
model = load_model('model_032300.h5')


def load_images(path, size=(256,512)):
    # load and resize the image
    pixels = load_img(path, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # split into satellite and map
    sat_img, map_img = pixels[:, :256], pixels[:, 256:]
    map_img = (map_img - 127.5) / 127.5
    map_img = expand_dims(map_img, 0)
    return map_img

def load_images_256(path, size=(256,256)):
    # load and resize the image
    pixels = load_img(path, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # split into satellite and map
    map_img = pixels[:, :]
    map_img = (map_img - 127.5) / 127.5
    map_img = expand_dims(map_img, 0)
    return map_img


# load source image
path = "Test_256/"
i = 0
dir_list = os.listdir(path)
for img in dir_list:
    inputname = 'Test_256/'+img
    src_image = load_images_256('Test_256/'+img)
    gen_image = model.predict(src_image)
    gen_image = (gen_image + 1) / 2.0
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    outputname = 'result/result_' + str(i) + img
    pyplot.savefig(outputname)
    pyplot.show()