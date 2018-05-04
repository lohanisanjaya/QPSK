import os
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

class Simulate_Homodyne(object):

    def __init__(self,alpha,beta):
        self.alpha = alpha  #signal strength 
        self.beta = beta    #LO strength 

    def Scale(self,data):
        return pp.scale(data)

    def Interference(self,A,B):
        return 2*self.alpha*self.beta*tf.sin(A-B)

    def Homodyne_without_noise(self,phi_signal=float,phi_LO =[0,2*np.pi],phi_LO_grid=800):
        phi_b = np.linspace(phi_LO[0],phi_LO[1],phi_LO_grid)
        Inter = []
        for i in range(len(phi_b)):
            Inter.append(self.Interference(phi_signal,phi_b[i]))
        return Inter


    def Homodyne_with_noise(self,phi_signal=np.pi/4,phi_LO = [0,2*np.pi],trials = 1000,\
                            phi_LO_grid=800):
        inten_mean = self.Homodyne_without_noise(phi_signal=phi_signal,phi_LO=[0,2*np.pi],\
                           phi_LO_grid=phi_LO_grid)        
        out_trials = []
        for j in range(len(inten_mean)):
            out_trials.append(tf.random_normal(shape=[trials],mean=inten_mean[j],stddev=self.beta,dtype=tf.float64))
        return out_trials #[[.... 1000] .. ..[] .. ..800]
