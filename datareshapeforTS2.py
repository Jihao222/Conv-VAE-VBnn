import numpy as np
from matplotlib import pyplot as plt
import os
#filename='D:/blowout-981/blowout-981/oldtest/series1.txt'
filename='D:/blowout981/oldtest/series1.txt'
aa=np.loadtxt(filename)
print(aa.shape)
bb=aa.reshape(6,79,72,110)
bb=np.array(bb)
print(bb.shape)
bb[bb<0.01]=0

for i in range(0,80):
    #plt.ion()
    cc=bb[1,i,:,:]
    print(cc)
    plt.imshow(cc)
    ax=plt.gca()
    ax.invert_yaxis()
    plt.show()
    plt.close()
    