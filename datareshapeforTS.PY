import numpy as np
from matplotlib import pyplot as plt
import os
from keras import backend as K
filename='D:/blowout981/oldtest/txt2'
filelist=os.listdir(filename)
cc=[]
for item in filelist:
    
    aa=np.loadtxt(os.path.join(filename,item))
    print(aa.shape)
    #bb=aa.reshape(81,72,97,110)
    bb=aa.reshape(1,72,97,110)
    print(bb.shape)
    bb=bb[1:1:1,:,38,:]
    print(bb.shape)
    shape = K.int_shape(bb)
    print(shape)
    print(bb.shape)
    cc.append(bb)
    #cc=np.array(bb).reshape(shape[0] * shape[1],shape[2])
    #np.savetxt('series1.txt',cc)
print(np.array(cc).shape)
cc=np.array(cc).reshape(34128,110)
np.savetxt('series1.txt',cc)