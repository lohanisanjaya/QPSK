import numpy as np
import os
import sys
import _pickle as pkl
from sklearn import preprocessing as pp


def read_pkl(filename):
    f = open(filename,'rb')
    g = pkl.load(f)
    f.close()
    return g

def data_reader(keys_size = 2,train_size=800,test_size=100):
    train_data = []
    test_data = []
    for i in range(keys_size):
        f = read_pkl('keys_{}.pkl'.format(i+1))
        #assert len(f) == train_size+test_size,'Total no. data != train+test'
        for j in range(train_size):
            train_data.append(f[j])
        for k in range(test_size):
            test_data.append(f[-(k+1)])
        
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    data_in_set = []
    data_out_set = []
    test_data_in_set = []
    test_data_out_set = []
    for i in range(len(train_data)):
        #data_in = pp.scale(data_file[i][0])
        #data_out = pp.scale(data_file[i][1])
        data_in = train_data[i][0]
        data_out = train_data[i][1]
        data_in_set.append(data_in)
        data_out_set.append(data_out)
    for i in range(len(test_data)):
        #data_in = pp.scale(data_file[i][0])
        #data_out = pp.scale(data_file[i][1])
        data_in = test_data[i][0]
        data_out = test_data[i][1]
        test_data_in_set.append(data_in)
        test_data_out_set.append(data_out)
   
    
    train_in,test_in = [np.array(data_in_set),np.array(test_data_in_set)]#<-----
    ###--------------------------------------------------------
    ####Reversing the direction, predicting inputs from outputs
    #train_in,test_in = np.array(data_out_set[:290]),\
    #                     np.array(data_out_set[290:])
    ###---------------------------------------------------------
    train_out,test_out = [np.array(data_out_set),np.array(test_data_out_set)]   #<-----------
    #---------------------------------------------------------------------------
    ###Reversing the direction, predicting inputs from outputs
    #train_out,test_out = np.array(data_in_set[:290]),np.array(data_in_set[290:])
    ###-------------------------------------------------------------------------
    return [train_in.flatten(),train_out.flatten(),test_in.flatten(),test_out.flatten()]

#def data_loader(filename):
#    data_in_flat,data_out_flat = data_reader(filename)
    
    
def data_reader_test_portion(filename,upto=36):
    test = read_pkl(filename)
    testX = []
    testY = []
    for i in range(len(test)):
        testX.append(test[i][0])
        testY.append(test[i][1])
    for j in range(len(testX)):
        testX[j][upto:] = 0
    return [np.array(testX).flatten(),np.array(testY).flatten()]     
    
