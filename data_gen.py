import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from sklearn import preprocessing as pp
import main_Homo as m
import random
import _pickle as pkl
import sys
upto_x = int(sys.argv[1])

n = int(sys.argv[2])


def norm(data,minm=0,maxm=1):
    data = (data - np.min(data))/(np.max(data)-np.min(data))
    data = data*(maxm-minm)+minm
    return data

homo = m.Simulate_Homodyne(alpha=1,beta=100**2)
sess = tf.Session()

keys = [np.pi/4,3*np.pi/4,5*np.pi/4,7*np.pi/4]
#keys = [np.pi/8,3*np.pi/8,5*np.pi/8,7*np.pi/8,9*np.pi/8,11*np.pi/8,13*np.pi/8,15*np.pi/8]
#keys = [np.pi/4,5*np.pi/4]
#####plotting only interference:
c = ['b','r','g','k']
#for i in range(len(keys)):
#    y1 = homo.Homodyne_without_noise(phi_signal=keys[i],phi_LO=[0,2*np.pi],phi_LO_grid=800)
#    plt.plot(np.linspace(0,2*np.pi,800),sess.run(y1),c=c[i])
#plt.show()

#####plotting homodyne outputs:

#for k in []:
#k = 0
for j in range(len(keys)):
    train_data = []
    y = homo.Homodyne_with_noise(phi_signal=keys[j],phi_LO = [0,n*np.pi/10],trials = 1000,\
                            phi_LO_grid=upto_x)
    y = sess.run(y)
    for k in range(500): #no of training data per key
        #one_hot = sess.run(tf.one_hot(j,4))
        one_hot = np.zeros(4)
        #one_hot = np.zeros(2)
        one_hot[j] = 1.
        means_homo = norm([random.choice(i) for i in y])
        train_data.append([means_homo,one_hot])
    f = open('keys_{}.pkl'.format(j+1),'wb')
    #f = open('keys_{}_test.pkl'.format(j+1),'wb')
    pkl.dump(train_data,f,-1)
    f.close()
    ##means_homo = pp.scale(np.array(means_homo))
#    plt.plot(np.linspace(0,2*np.pi,720),means_homo,c=c[j],label='key: {}'.format(keys[j]))
    ##y_scaled = np.array(sess.run(y)).flatten()  #pp.scale(np.array(sess.run(y)).flatten())
    ##plt.scatter(np.repeat(np.linspace(0,2*np.pi,1000),1000),y_scaled,s=0.001,c=c[j])
#    plt.xlabel('LO phase')
#    plt.ylabel('Intensity') 
#    plt.legend()
#plt.savefig('one_homodyne_2_3.png'.format(k))
#plt.show()


