'''Example of VAE on MNIST dataset using CNN

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import cv2
import os 
from skimage import data_dir,io,transform,color
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import keras
import seaborn as sns
import tensorflow as tf
import random
import time
from sklearn.model_selection import train_test_split
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
random.seed(10)
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    #
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
time1=time.time()
print(time.time())
class Config:
    MODEL_PATH = "D:/blowout-981/blowout-981/oldtest/vae_cnn_mnist.h5"
    #MODEL_PATH2 = "C:/Users/18953/Desktop/MODEL/新建文件夹 (3)/mymodel4.hdf5"
datafile='D:/blowout-981/blowout-981/oldtest/180_360.txt'
aa=np.loadtxt(datafile)
print(aa.shape)

x_train=aa.reshape(72,72,97)
#ts=np.array([x_train[54],x_train[31],x_train[18],x_train[2],x_train[16],x_train[64],x_train[27],x_train[45],x_train[43],x_train[23],x_train[65],x_train[7],x_train[48],x_train[20],x_train[14]])
#np.savetxt('D:/blowout-981/blowout-981/oldtest/x_original.txt',ts.reshape(104760))
# print(ts.shape)
# for i in range(15):
    # ax = plt.subplot(1, 2, 1)
    # levels=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    # h=plt.contourf(ts[i],levels,cmap='jet')
    # f3=plt.colorbar(h)
    # plt.show()
x_train=np.array(x_train)

maximum=np.max(x_train)
print(maximum)

plt.show()
x_train = x_train.astype('float32')/maximum
############
datafile2='D:/blowout-981/blowout-981/oldtest/aa_test.txt'
bb=np.loadtxt(datafile2)
print(bb.shape)
x_test=bb.reshape(15,72,97)
x_test=np.array(x_test)
x_test = x_test.astype('float32') 
print((x_test).shape)
#####################
for i in range(15):
    ax = plt.subplot(1, 2, 1)
    levels=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    
    h=plt.contourf(x_train[i+16]*maximum,levels,cmap='jet')

    f3=plt.colorbar(h)
    f3.set_label(label='Concentration',size=22,weight='bold')
    f3.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
#############


print('2222222222222222222222222222')
image_size = x_train.shape[1]
image_size2= x_train.shape[2]
print(image_size,image_size2)
x_train = np.reshape(x_train, [-1, image_size, image_size2, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size2, 1])
print(x_test.shape)
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255


# network parameters
input_shape = (image_size, image_size2, 1)
batch_size = 12
kernel_size = 3
filters = 32
latent_dim = 2
epochs = 300

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x=Conv2D(filters=16,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
x=Conv2D(filters=32,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=BatchNormalization()(x)
x=Conv2D(filters=64,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=BatchNormalization()(x)
#x=Conv2D(filters=128,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=Conv2D(filters=8,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=BatchNormalization()(x)
# for i in range(2):
    # filters *= 2
    # x = Conv2D(filters=filters,
               # kernel_size=kernel_size,
               # activation='relu',
               # strides=1,
               # padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

print(shape[0])
print(shape)
# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

#x=Conv2DTranspose(filters=8,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=Conv2DTranspose(filters=128,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=BatchNormalization()(x)
x=Conv2DTranspose(filters=64,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=BatchNormalization()(x)
x=Conv2DTranspose(filters=32,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
x=Conv2DTranspose(filters=16,kernel_size=kernel_size,activation='relu',strides=1,padding='same')(x)
#x=BatchNormalization()(x)
# for i in range(2):
    # x = Conv2DTranspose(filters=filters,
                        # kernel_size=kernel_size,
                        # activation='relu',
                        # strides=1,
                        # padding='same')(x)
    # filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_)
    args = parser.parse_args()
    print(args)
    models = (encoder, decoder)
    data = (x_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        
       reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs)) 
        
    else:
        
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))
        

    def vae_loss(x, t_decoded):
    #'''Total loss for the plain VAE'''
        return K.mean(reconstruction_loss(x, t_decoded) + vae_kl_loss)


    reconstruction_loss *= image_size * image_size2
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    #vae.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    if args.weights:
    ####test the uncertainty#####
        vae.load_weights(args.weights)
        
        ####test the specific leak rate=80 wind speed=0 and wind direction=360####
        # z_sample1=[[-3.220,0.204]]
        #z_sample1=[[-4.25,-0.304]]
        #z_sample1=[[-4.535066127777099609e+00,3.775246143341064453e-01]]
        # z_sample1=np.array(z_sample1)
        # x_decoded = decoder.predict(z_sample1)*maximum
        # x_decoded[x_decoded<0.01]=0
        # levels=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
  
        # h=plt.contourf(x_decoded.reshape(image_size,image_size2),levels,cmap='jet')
        # f3=plt.colorbar(h)
        # plt.show()
        ##########################################################
        
        testfile0='D:/blowout-981/blowout-981/oldtest/y_mean50.txt'
        y_mean1=np.loadtxt(testfile0)
        x_decoded2=[]
        #print(y_mean0[0,:])
        for i in range(500):
            y_mean0=y_mean1[:,i]
            
            y_mean0=y_mean0.reshape(12,2)
            #print(y_mean0)
            z_sample1=y_mean0[5,:].reshape(1,2)
            #print(y_mean0[2,:])
            #z_sample1=[[1.182253956794738770e-01, -1.197193741798400879e+00]]
            z_sample1=np.array(z_sample1)
            x_decoded = decoder.predict(z_sample1)*maximum
            x_decoded[x_decoded<0.01]=0
            x_decoded2=np.append(x_decoded2,x_decoded)
        #x_decoded2=x_decoded2.reshape(3*image_size*image_size2)
        
        x_decoded2=x_decoded2.reshape(500,image_size*image_size2)
        x_decoded2_mean = np.mean(x_decoded2, axis=0)
        #x_decoded2_mean[x_decoded2_mean<0.01]=0


        x_decoded2_sigma = np.std(x_decoded2, axis=0)*3
        #print(np.mean(x_decoded2_mean-x_decoded2_sigma))
        abc=x_decoded2_mean+x_decoded2_sigma 
        abc2=x_decoded2_mean-x_decoded2_sigma
        #print(x_decoded2_mean.shape)
        ax = plt.subplot(1, 2, 1)
        
        
        levels=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
        # h=plt.contourf(x_decoded2_mean.reshape(image_size,image_size2),levels,cmap='jet')
        # h.ax.set_ylabel('Y direction',size=16)
        # h.ax.set_xlabel('X direction',size=16)
        # #f3=plt.colorbar(h,label="Concentration")
        # #f3.ax.tick_params(labelsize=16)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        
        
        
        #ax = plt.subplot(1, 3, 2 )
        h=plt.contourf(x_decoded2_mean.reshape(image_size,image_size2),levels,cmap='jet')
        
        f3=plt.colorbar(h)
        f3.set_label(label='Concentration',size=22,weight='bold')
        f3.ax.tick_params(labelsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        levels=np.linspace(0.0025,0.0575, 10)
        #levels=[0.0025,0.0040,0.0055,0.0070,0.0085,0.0100,0.0125,0.0175]
        #levels=[0.0025:0.0175:10]
        ax = plt.subplot(1,2, 2 )     
        h=plt.contourf(x_decoded2_sigma.reshape(image_size,image_size2),levels,cmap='jet')
        f3=plt.colorbar(h)
        f3.set_label(label='Uncertainty',size=22,weight='bold')
        f3.ax.tick_params(labelsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        #plt.show()
        time2=time.time()
        
        print(time2-time1)
        #cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小
        # ax=plt.gca()
        # ax.invert_yaxis()
        # ax.get_xaxis().set_visible(True)
        # ax.get_yaxis().set_visible(True)
        
        # levels=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
  
        # h=plt.contourf(x_decoded.reshape(image_size,image_size2),levels,cmap='jet')
        # f3=plt.colorbar(h)
        # plt.show()
        
        # testfile0='D:/blowout-981/blowout-981/oldtest/y_mean0.txt'
        # testfile1='D:/blowout-981/blowout-981/oldtest/y_mean1.txt'
        # testfile2='D:/blowout-981/blowout-981/oldtest/y_mean2.txt'
        # testfile3='D:/blowout-981/blowout-981/oldtest/y_mean3.txt'
        # y_mean0=np.loadtxt(testfile0)
        # z_sample=[]
        # y_mean1=np.loadtxt(testfile1)
        # y_mean2=np.loadtxt(testfile2)
        # y_mean3=np.loadtxt(testfile3)
        # for j in range(500):
            # for i in range(12):
                # #print(y_mean0[i,:])
                # ss = np.array((y_mean0[i,j], y_mean1[i,j],y_mean2[i,j],y_mean3[i,j]))
                # z_sample=np.append(z_sample,ss)
                # print(z_sample.shape) 
           
        # x_decoded2 =[]
        # # z_sample=y_mean0
        # z_sample=z_sample.reshape(500,12,4)
        # print(z_sample.shape)
        # for i in range(500):
            # z_sample1=z_sample[i,5,:].reshape(1,latent_dim) 
            # #print(z_sample1.shape)
            # x_decoded = decoder.predict(z_sample1)*maximum
            # x_decoded[x_decoded<0.01]=0
            # #print(x_decoded.shape)
            # x_decoded2=np.append(x_decoded2,x_decoded)
            # #x_decoded2[x_decoded2<0.011]=0
            

        # x_decoded2=x_decoded2.reshape(500*image_size*image_size2)
        
        # x_decoded2=x_decoded2.reshape(500,image_size*image_size2)
        
        # x_decoded2_mean = np.mean(x_decoded2, axis=0)
        # x_decoded2_mean[x_decoded2_mean<0.01]=0


        # x_decoded2_sigma = np.std(x_decoded2, axis=0)*3
        # print(np.mean(x_decoded2_mean-x_decoded2_sigma))
        # abc=x_decoded2_mean+x_decoded2_sigma 
        # abc2=x_decoded2_mean-x_decoded2_sigma
        # print(x_decoded2_mean.shape)
        # ax = plt.subplot(1, 2, 1)
        
        
        # levels=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
        # # h=plt.contourf(x_decoded2_mean.reshape(image_size,image_size2),levels,cmap='jet')
        # # h.ax.set_ylabel('Y direction',size=16)
        # # h.ax.set_xlabel('X direction',size=16)
        # # #f3=plt.colorbar(h,label="Concentration")
        # # #f3.ax.tick_params(labelsize=16)
        # # plt.xticks(fontsize=16)
        # # plt.yticks(fontsize=16)
        
        
        
        # #ax = plt.subplot(1, 3, 2 )
        # h=plt.contourf(x_decoded2_mean.reshape(image_size,image_size2),levels,cmap='jet')
        
        # f3=plt.colorbar(h)
        # f3.set_label(label='Concentration',size=22,weight='bold')
        # f3.ax.tick_params(labelsize=16)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        
        # levels=np.linspace(0.0025,0.0175, 10)
        # #levels=[0.0025,0.0040,0.0055,0.0070,0.0085,0.0100,0.0125,0.0175]
        # #levels=[0.0025:0.0175:10]
        # ax = plt.subplot(1,2, 2 )     
        # h=plt.contourf(x_decoded2_sigma.reshape(image_size,image_size2),levels,cmap='jet')
        # f3=plt.colorbar(h)
        # f3.set_label(label='Uncertainty',size=22,weight='bold')
        # f3.ax.tick_params(labelsize=16)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # #cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小
        # # ax=plt.gca()
        # # ax.invert_yaxis()
        # # ax.get_xaxis().set_visible(True)
        # # ax.get_yaxis().set_visible(True)
        
        
        
        
        
        # plt.show()
        # print(x_decoded2_mean.shape)
      
    
    else:
        # train the autoencoder
        checkpoint = ModelCheckpoint(filepath=Config.MODEL_PATH,verbose=1,monitor='val_loss', save_weights_only=False,mode='auto' ,save_best_only=True,period=1)
        history=vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test,None),callbacks=[checkpoint]).history
        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        #ax.plot(history['loss'], 'b', label='Train', linewidth=2)
        ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
        ax.set_title('Model loss', fontsize=16)
        ax.set_ylabel('Loss (mae)')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        #vae.save_weights('vae_cnn_mnist.h5')
        plt.show()
        
############################








        
###########






#############################################
#####show 6 figures of varied wind speeds from 0 to 12 m/s.####
cc=[]
cc1=[]
for i in range(15): 
    _, _, z=encoder.predict(x_test[i].reshape(-1,image_size, image_size2,1))
    print(z.shape)
    cc.append(z)
for i in range(72): 
    _, _, z1=encoder.predict(x_train[i].reshape(-1,image_size, image_size2,1))
    print(z1.shape)
    cc1.append(z1)   

# xx=vae.predict(x_test[i].reshape(-1,image_size, image_size2,1))*maximum
# print(xx.shape)
# plt.imshow(xx.reshape(image_size, image_size2))
# ax=plt.gca()
# ax.invert_yaxis()
# plt.show()
# plt.imshow((x_test[i]*maximum).reshape(image_size, image_size2))
# ax=plt.gca()
# ax.invert_yaxis()
# #plt.savefig(xx[1])
# plt.show()

##we'll plot 10 images.
def compute_mae_mse_rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    mae=sum(absError)/len(absError)  # 平均绝对误差MAE
    mse=sum(squaredError)/len(squaredError)  # 均方误差MSE
    RMSE=sum(absError)/len(absError)
    return mae,mse,RMSE
n = 4
plt.figure(figsize=(5, 5))
outputs2=vae.predict(x_test.reshape(-1,image_size, image_size2,1))*maximum
reconstruction_loss = compute_mae_mse_rmse((x_test.reshape(104760))*maximum, outputs2.reshape(104760))  
print(reconstruction_loss)
for i in range(n):
    # Show the originals
    xx=vae.predict(x_test[i].reshape(-1,image_size, image_size2,1))*maximum
    print(xx.shape)

    ax = plt.subplot(2, n, i + 1)
    levels=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
  
    h=plt.contourf(xx.reshape(image_size,image_size2),levels,cmap='jet')
    
    #f3=plt.colorbar(h)
    #f3.set_label(label='Concentration',size=22,weight='bold')
    #f3.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)






    
    # plt.imshow(xx.reshape(image_size, image_size2))
    # ax=plt.gca()
    # ax.invert_yaxis()
    # ax.get_xaxis().set_visible(True)
    # ax.get_yaxis().set_visible(True)
   
    ax = plt.subplot(2, n, i + 1 + n)
    hh=plt.contourf((x_test[i]*maximum).reshape(image_size,image_size2),levels,cmap='jet')
    
    #f33=plt.colorbar(hh)
    #f33.set_label(label='Concentration',size=22,weight='bold')
    #f33.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
      
plt.show()    

print(np.array(cc).reshape(15,latent_dim))
cc=np.array(cc).reshape(15,latent_dim)
np.savetxt('D:/blowout-981/blowout-981/oldtest/z0.txt',cc)     
cc1=np.array(cc1).reshape(72,latent_dim)
np.savetxt('D:/blowout-981/blowout-981/oldtest/z1.txt',cc1)  
# plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")

    
# encoded_imgs = vae.predict(encoder(x_test)[2])
# n = 10 

# for i in range(1,n):
    # plt.figure(figsize=(20, 2))
    # ax = plt.subplot(1, n, i)
    # plt.imshow(encoded_imgs[i].reshape(28, 28))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # plt.show()
