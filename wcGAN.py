import cv2
from numpy import load
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from numpy import savez_compressed
from keras.models import load_model
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import keras.backend as K

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

# Discrminiator
def disc(image_shape):

	init = RandomNormal(stddev=0.02)
	source_img = Input(shape=image_shape)
	target_img = Input(shape=image_shape)
	merged = Concatenate()([source_img, target_img])

	x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(x)
	patch_out = Activation('sigmoid')(x)
	model = Model([source_img, target_img], patch_out)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=wasserstein_loss, optimizer=opt, loss_weights=[0.5])
	return model


def encoder(layer_in, n_filters, batchnorm=True):
	init = RandomNormal(stddev=0.02)
	x = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		x = BatchNormalization()(x, training=True)
	x = LeakyReLU(alpha=0.2)(x)
	return x

def decoder(layer_in, skip_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	x = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	x = BatchNormalization()(x, training=True)
	if dropout:
		x = Dropout(0.5)(x, training=True)
	x = Concatenate()([x, skip_in])
	x = Activation('relu')(x)
	return x

# Generator
def generator(image_shape=(256,256,3)):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	e1 = encoder(in_image, 64, batchnorm=False)
	e2 = encoder(e1, 128)
	e3 = encoder(e2, 256)
	e4 = encoder(e3, 512)
	e5 = encoder(e4, 512)
	e6 = encoder(e5, 512)
	e7 = encoder(e6, 512)
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	d1 = decoder(b, e7, 512)
	d2 = decoder(d1, e6, 512)
	d3 = decoder(d2, e5, 512)
	d4 = decoder(d3, e4, 512, dropout=False)
	d5 = decoder(d4, e3, 256, dropout=False)
	d6 = decoder(d5, e2, 128, dropout=False)
	d7 = decoder(d6, e1, 64, dropout=False)
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	model = Model(in_image, out_image)
	return model

#  combined generator and discriminator model
def GAN(gen_model, disc_model, image_shape):
	for layer in disc_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	in_src = Input(shape=image_shape)
	gen_out = gen_model(in_src)
	dis_out = disc_model([in_src, gen_out])
	model = Model(in_src, [dis_out, gen_out])
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=[wasserstein_loss, 'mae'], optimizer=opt, metrics=['accuracy'], loss_weights=[1,100])
	return model


def load_real_images(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def generate_real_images(dataset, n_images, patch_shape):
	trainA, trainB = dataset
	ix = randint(0, trainA.shape[0], n_images)
	X1, X2 = trainA[ix], trainB[ix]
	y = ones((n_images, patch_shape, patch_shape, 1))
	return [X1, X2], y

def generate_fake_images(gen_model, images, patch_shape):
	X = gen_model.predict(images)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def stats(step, gen_model, dataset, n_images=3):
	[X_realA, X_realB], _ = generate_real_images(dataset, n_images, 1)
	X_fakeB, _ = generate_fake_images(gen_model, X_realA, 1)
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_images):
		pyplot.subplot(3, n_images, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_images):
		pyplot.subplot(3, n_images, 1 + n_images + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_images):
		pyplot.subplot(3, n_images, 1 + n_images*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])

	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	gen_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# training
def train(disc_model, gen_model, gan_model, dataset, n_epochs=100, n_batch=1):
	n_patch = disc_model.output_shape[1]
	trainA, trainB = dataset
	bat_per_epo = int(len(trainA) / n_batch)
	n_steps = bat_per_epo * n_epochs
	for i in range(n_steps):
		[X_realA, X_realB], y_real = generate_real_images(dataset, n_batch, n_patch)
		X_fakeB, y_fake = generate_fake_images(gen_model, X_realA, n_patch)
		d_loss1 = disc_model.train_on_batch([X_realA, X_realB], y_real)
		d_loss2 = disc_model.train_on_batch([X_realA, X_fakeB], y_fake)
		g_loss,_,_,_,_ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		if (i+1) % (bat_per_epo * 10) == 0:
			stats(i, gen_model, dataset)

def load_data(path, size=(256,512)):
    source, target = list(), list()
    for filename in listdir(path):
        data = load_img(path + filename, target_size=size)
        data = img_to_array(data)
        sat_img, map_img = data[:, :256], data[:, 256:]
        source.append(sat_img)
        target.append(map_img)
    return [asarray(source), asarray(target)]
 
path = 'FinalDataset/'
# path="/content/drive/MyDrive/CS639/FinalDataset/"
[tar_imgs, source_imgs] = load_data(path)
print('Loaded: ', source_imgs.shape, tar_imgs.shape)
filename = 'pairs_256.npz'
savez_compressed(filename, source_imgs, tar_imgs)
print('Saved dataset: ', filename)

# load the dataset
data = load('pairs_256.npz')
source_imgs, tar_imgs = data['arr_0'], data['arr_1']
print('Loaded: ', source_imgs.shape, tar_imgs.shape)

# plot source images
n_images = 3
for i in range(n_images):
 pyplot.subplot(2, n_images, 1 + i)
 pyplot.axis('off')
 pyplot.imshow(source_imgs[i].astype('uint8'))

# plot target image
for i in range(n_images):
 pyplot.subplot(2, n_images, 1 + n_images + i)
 pyplot.axis('off')
 pyplot.imshow(tar_imgs[i].astype('uint8'))
pyplot.show()

dataset = load_real_images('pairs_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
disc_model = disc(image_shape)
gen_model = generator(image_shape)
gan_model = GAN(gen_model, disc_model, image_shape)
train(disc_model, gen_model, gan_model, dataset)
