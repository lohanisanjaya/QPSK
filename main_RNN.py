'''Correspondence:
   SL (slohani@tulane.edu)
   RTG (rglasser@tulane.edu)
   Feb 17, 2018'''

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
#import cPickle as pkl
import _pickle as pkl
import time
import data_loader as dl
import sys
class Local_Rnn_LSTM(object):

        def __init__(self, eta = 0.001, epochs = 20, batch_size = 5, time_steps = 6,\
                          num_inputs = 5, hidden_units = 20, num_classes = 10):

            #Training Parameters
            self.eta = eta
            self.epochs = epochs	#Generally Iterations = epochs*batch_size
            self.batch_size = batch_size 
            self.nor = 0.
            #Network Parametes
            #data are fed to LSTM row-wise
            self.time_steps = time_steps #number of rows data
            self.num_inputs = num_inputs #number of columns data
            self.hidden_units = hidden_units #number of hidden neurons in ANN, dimension of weight
            self.num_classes = num_classes   #number of output neurons

        def Pkl_Read(self,filename):
            file_open = open(filename,'rb')
            value = pkl.load(file_open)
            return value

        def norm(self,data,minm=0,maxm=255):
            data = (data - np.min(data))/(np.max(data)-np.min(data))
            data = data*(maxm-minm)+minm
            return data 

        def Pkl_Save(self,filename,value):
            file_open = open(filename,'wb')
            return pkl.dump(value,filename,protocol=pkl.HIGHEST_PROTOCOL)

        def Data_Loader(self,data_file = 'g_pulse.pkl'):
            ''' It creats an iterator of training sets with respect to batch_size. We can call
                this using next(iterator) command while running the optimization.
                The data should be list of [[array_X.flatten],[labels_Y]]. For eg, if you want to have 10
                training images (128x128 pixs) with labels ranging from 0 to 9 then 
                                   data = X,Y = [[array_flatten[0, .. .. ..]_128x128,
                                           array[0., 0.. .. ..]_128x128,
                                           array[0, .. .. ..]_128x128,
                                            .. ... .... ... ... ..], array[0,1,2,3,4,5,6,7,8,9]]
                For debuging read at https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/ '''
             
            data_X_in,data_Y_out,test_in,test_out = dl.data_reader(data_file)
            X_seq = [np.array(data_X_in[i * self.num_inputs: (i + 1) * self.num_inputs]) for i\
                      in range(len(data_X_in) // self.num_inputs)]
            Y_seq = [np.array(data_Y_out[i * self.num_classes: (i + 1) * self.num_classes])for i\
                       in range(len(data_Y_out) // self.num_classes)]
            X_in = np.array([X_seq[i*self.time_steps:(i +1)*self.time_steps] for i in range(int(len(X_seq)//self.time_steps))])
            Y_out = np.array(Y_seq) #np.array([Y_seq[i + self.time_steps] for i in range(len(Y_seq)\
                          #- self.time_steps)])
         #   data_X,data_Y = self.Pkl_Read(filename = data_file)
         #   self.nor = np.linalg.norm(data_X)
         #   data_X = data_X/self.nor
         #   self.data_X = data_X
         #   print ('norm', self.nor)
            #assert len(data_X[0]) == (self.time_steps*self.num_inputs), 'Time steps x Num_inputs \
            #                                           does not match with dimension of input data'
            #*#For time-series
            #assert len(data_X) == (self.time_steps*self.num_inputs), 'Time steps x Num_inputs \
            #                                           does not match with dimension of input data'
            #data_X = data_X.reshape(self.batch_size,self.time_steps,self.num_inputs)
         #   data_X_seq = [np.array(data_X[i * self.num_inputs: (i + 1) * self.num_inputs]) for i\
         #              in range(len(data_X) // self.num_inputs)]
         #   data_X_out = np.array([data_X_seq[i: i + self.time_steps] for i in range(len(data_X_seq\
         #               )- self.time_steps)])
         #   data_Y_out = np.array([data_X_seq[i + self.time_steps] for i in range(len(data_X_seq)\
         #                 - self.time_steps)]) #[0] for only one output prediction/ next element in seq
         #   print ('X {}, shape {}'.format(data_X_out*self.nor ,data_X_out.shape))
         #   print ('Y',data_Y_out*self.nor)

            #------------------------------------------------------------
            #*#For one_hot type outputs <----------------
            ##shuffle the data
            #zipped = list(zip(data_X,data_Y))
            #np.random.shuffle(zipped)
            #data_X,data_Y = list(zip(*zipped))
            ###X_batch = np.array(data_X).reshape(self.batch_size,self.time_steps,self.num_inputs)
            ###generating one hot form for Y_batch
            #indices = np.array(data_Y)
            #depth = self.num_classes
            #data_Y = tf.one_hot(indices,depth) #Dimension of len(data_Y) X num_classes 
            #----------------------------------------------------------------------------------
         #   return [data_X,data_X_out,data_Y_out] #data_X is a list of [number of training sets X (time_steps*num_inputs)]
            return [X_in,Y_out,test_in,test_out]
        def LSTM(self):
            return rnn.BasicLSTMCell(self.hidden_units,forget_bias=1.0)

        def Build_BasicLSTM(self,X_data,weights,biases,time_steps):
            #current data input shape is (batch_size,time_steps,num_inputs)
            X = tf.unstack(X_data,time_steps,axis=1)
            ####X = X_data
            #LSTM_CELL = rnn.BasicLSTMCell(self.hidden_units,forget_bias=1.0)
            ####LSTM_CELL = rnn.LSTMCell(self.hidden_units)
            Multi_LSTM_CELL = rnn.MultiRNNCell([self.LSTM() for _ in range(1)],state_is_tuple=True)
            outs,states = tf.nn.static_rnn(Multi_LSTM_CELL,X,dtype=tf.float32)
            #outs,states = tf.nn.static_rnn(LSTM_CELL,X,dtype=tf.float32)
            #print (sess.run(outs[-1].shape))          
            return tf.matmul(outs[-1],weights['final'])+biases['final']
        
        def Build_DNN(self,X,X_len = 720,hidden_neurons = 1000,out_neurons=720):
            w1 = tf.Variable(tf.truncated_normal([X_len,hidden_neurons]))
            b1 = tf.Variable(tf.zeros([hidden_neurons]))
            w2 = tf.Variable(tf.truncated_normal([hidden_neurons,hidden_neurons]))
            b2 = tf.Variable(tf.zeros([hidden_neurons]))
            #w_2 = tf.Variable(tf.truncated_normal([hidden_neurons,hidden_neurons]))
            #b_2 = tf.Variable(tf.zeros([hidden_neurons]))
            #w__2 = tf.Variable(tf.truncated_normal([hidden_neurons,hidden_neurons]))
            #b__2 = tf.Variable(tf.zeros([hidden_neurons]))
            w3 = tf.Variable(tf.truncated_normal([hidden_neurons,out_neurons]))
            b3 = tf.Variable(tf.zeros([out_neurons]))
           
            first_hidden = tf.nn.sigmoid(tf.matmul(tf.reshape(X,[-1,X_len]),w1)+b1)
            snd_hidden = tf.nn.sigmoid(tf.matmul(first_hidden,w2)+b2)
            #thd_hidden = tf.nn.sigmoid(tf.matmul(snd_hidden,w_2)+b_2)
            #fth_hidden = tf.nn.sigmoid(tf.matmul(thd_hidden,w__2)+b__2)

            outputs = tf.matmul(snd_hidden,w3)+b3
            return outputs

        def Build_Regression(self,X,X_len,out_neurons=400):
            w = tf.Variable(tf.truncated_normal([X_len,out_neurons]))
            b = tf.Variable(tf.zeros([out_neurons]))
            return tf.matmul(tf.reshape(X,[-1,X_len]),w)+b

        def Run_LSTM_DNN(self,keys_size = 4,Network=None):
            start = time.time()
            #--------------------------------------------------------------------------
            X = tf.placeholder(tf.float32,[None,self.time_steps,self.num_inputs],name='X')
            Y = tf.placeholder(tf.float32,[None,self.num_classes],name='Y')
            #--------------------------------------------------------------------------
            if Network == 'LSTM':
            ###--------------------LSTM SETUP---------------------------------------------
                weights = {'final': tf.Variable(tf.random_normal([self.hidden_units,\
                       self.num_classes]))}
                biases = {'final': tf.Variable(tf.random_normal([self.num_classes]))}
                logits = self.Build_BasicLSTM(X,weights,biases,time_steps=self.time_steps)
                Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits\
                                          = logits, labels = Y))
                prediction_series = tf.nn.softmax(logits)
                
                #prediction_series = logits
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            elif Network == 'DNN':
            ###--------------------DNN SETUP------------------------------------------------
                outputs = self.Build_DNN(X,X_len = 720,hidden_neurons = 800,out_neurons=4)
                #Loss = tf.reduce_mean(tf.square(outputs-Y))
                Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits\
                                          = outputs, labels = Y))
                prediction_series = tf.nn.softmax(outputs)
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #---------------------------------------------------------------------------       
          
            elif Network == 'LSTM_DNN':
            ###------------------LSTM+DNN SETUP-------------------------------------------
                weights = {'final': tf.Variable(tf.random_normal([self.hidden_units,\
                                self.num_classes]))}
                biases = {'final': tf.Variable(tf.random_normal([self.num_classes]))}
                logits = self.Build_BasicLSTM(X,weights,biases,time_steps=self.time_steps)
                outputs = self.Build_DNN(X=logits,X_len = self.num_classes,hidden_neurons = 20,\
                                     out_neurons=4)
                Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits\
                                          = outputs, labels = Y))
                prediction_series = tf.nn.softmax(outputs)
                #Loss = tf.reduce_mean(tf.square(outputs-Y))
                #prediction_series = outputs
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            elif Network == 'Regression':
            ###-------------------------Regression-----------------------------------------
                out_reg = self.Build_Regression(X,X_len=self.time_steps,out_neurons=4)
                Loss = tf.reduce_mean(tf.square(out_reg-Y))
                prediction_series = out_reg           
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            elif Network == 'DNN_LSTM':
            ###--------------------DNN+LSTM-------------------------------------------------
                outputs = self.Build_DNN(X,X_len = self.time_steps,hidden_neurons = 1500,out_neurons=4)
                #X_in = tf.reshape(outputs,[-1,self.time_steps,self.num_inputs])
                X_in = tf.reshape(outputs,[-1,4,self.num_inputs])
                weights = {'final': tf.Variable(tf.random_normal([self.hidden_units,\
                                self.num_classes]))}
                biases = {'final': tf.Variable(tf.random_normal([self.num_classes]))}
                logits = self.Build_BasicLSTM(X_in,weights,biases,time_steps=4)
                #Loss = tf.reduce_mean(tf.square(logits-Y))
                Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits\
                                          = logits, labels = Y))
                prediction_series = tf.nn.softmax(logits,name='predition_series')
                #prediction_series = logits
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
            else:
                print ('|Please define Network operation, select either of the followings')
                print ('|LSTM, DNN_LSTM, LSTM_DNN, Regression, DNN')
                print ('|contact: slohani@tulane.edu; rglasser@tulane.edu')
                sys.exit()
            #*#For time-series inputs #Y has [1] dimension <--------
            Optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.eta)
            #Optimizer = tf.train.AdagradOptimizer(learning_rate=self.eta)
            Training = Optimizer.minimize(Loss)
            #-----------------------------------------------------------------------
            #*#For softmax type output/one_hot kind output <---
            #------------------------------------------------------------------------------------
            #final_predictions = tf.nn.softmax(logits) 
            ##LOSS and OPTIMIZER
            #Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))
            #Optimizer = tf.train.AdagradOptimizer(learning_rate=self.eta)
            #Training = Optimizer.minimize(Loss)
            # Evaluate model (with test logits, for dropout to be disabled)
            correct_pred = tf.equal(tf.argmax(prediction_series, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='Accuracy')
            #--------------------------------------------------------------------------------------
            # Initialize the variables (i.e. assign their default value)
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            
            #Load the data
            X_data,Y_data,test_in,test_out = self.Data_Loader(keys_size)     
            #For only one_hot type data <-------           
            #Y_data = sess.run(Y_data)  #feed data should not be in tensor format
            #plot_g = []
            start = time.time()
            loss_list = []
            accu_plot = []
            accu_list_only = [0]
            #endtime = time.time()
            #diff = endtime-start
            #while diff < 240:
            for num_epoch in range(self.epochs):
                print ('')
                print ('|Epoch: {}'.format(num_epoch+1))
                print ('')
                gg = []
                for i in range(int(len(X_data)/self.batch_size)):
                  #for k in range(1):
                    #print (i)
                    #gg = []
                    #for j in range(30):
                #    i = 0
                    #gg = []
                    X_batch = np.array(X_data[i*self.batch_size:(i+1)*self.batch_size]).reshape(\
                              self.batch_size,self.time_steps,self.num_inputs)
                    Y_batch = Y_data[i*self.batch_size:(i+1)*self.batch_size].reshape(\
                              self.batch_size,self.num_classes)
                    
                    sess.run(Training,feed_dict={X:X_batch,Y:Y_batch})
                    #log,w,outp = sess.run([logits,weights,out],feed_dict={X:X_batch,Y:Y_batch})
                    #print ('logits',log.shape)
                    #print ('wt',w.get('final').shape)
                    #print ('out',outp.shape)
                    loss = sess.run([Loss],feed_dict={X:X_batch,Y:Y_batch})
                    print ('|Batch: {} Loss: {}'.format(i+1,loss))#<----
                    loss_list.append(loss)
                    #diff = time.time() - start
                    ##for series-data
                    #pred,loss = sess.run([prediction_series,Loss],feed_dict={X:X_batch,Y:Y_batch})
                    #gg.append(np.array(pred)*self.nor)
                    #print ('pred :{}'.format(np.array(pred)))
                test_X_in = test_in.reshape(self.batch_size,self.time_steps,self.num_inputs)
                test_Y_in = test_out.reshape(self.batch_size,self.num_classes)
                accu = sess.run([accuracy],feed_dict={X:test_X_in,Y:test_Y_in})
                print ('accuracy: {}'.format(accu))
                if accu[0] >= accu_list_only[-1]:
                    accu_plot.append(accu[0])
                else:
                    #accu_plot.append(None) #this will bypass the acc_fluctuation
                    accu_plot.append(accu[0])
                if accu[0] > np.max(accu_list_only):
                    saver.save(sess,'/home/slohani/Desktop/Dec_20_corrected/QPSK/pre_trained_net')
                else:
                    pass
                accu_list_only.append(accu[0])
  
            ###########Test_predictions
            #test_X_in = test_in.reshape(10,self.time_steps,self.num_inputs)
            #test_Y_in = test_out.reshape(10,self.time_steps,self.num_inputs)
            #pred = sess.run([prediction_series],feed_dict={X:test_X_in})
            #outfiles = [test_X_in,test_Y_in,pred,loss_list]
            #f = open('predicted.pkl','wb')
            #pkl.dump(outfiles,f,-1)
            #f.close()
            ##Test set
            ##test_X = self.Pkl_Read('data_2.pkl')
            ##test_X = test_X/self.nor
            ##test_X = test_X.reshape(self.batch_size,self.time_steps,self.num_inputs)
            ##test_X = np.array(self.data_X[300:330]).reshape(1,10,3)
            ##test_X = np.array(self.data_X[270:330]).reshape(1,10,6)
            ##predicted_values = []
            ##predicted_binary = []
            ##for k in range(5):
            ##    pred = sess.run([prediction_series],feed_dict={X:test_X})
            ##    pred_test_binary = np.array(pred)
            ##    print (pred_test_binary)
            ##    predicted_binary.append(pred_test_binary)
            ##    pred_appended = np.array(predicted_binary).flatten()
            ##    new_test = np.concatenate((np.array(self.data_X[276+6*k:330]),pred_appended),axis=0)
              ##  new_test = np.concatenate((np.array(self.data_X[60+3*k:330]),pred_appended),axis=0)
            ##    test_X = new_test.reshape(1,10,6)
            ##    predicted_values.append(np.array(pred)*self.nor)
            ##plot_g = np.concatenate((np.array(gg).flatten(),np.array(predicted_values).flatten()),\
            ##         axis=0)
            ##print ('plot_g',list(plot_g))   
            
          #  #plt.plot(range(60,360),np.array(plot_g))
            ##plt.plot(range(300,360),np.array(plot_g))
            ##plt.plot(range(330),true_X*self.nor)
            ##print (np.array(pred)*self.nor)     
            ##plt.show()
            #plt.plot(range(len(loss_list)),loss_list)
            print (accu_plot)
            #plt.plot(range(1,len(accu_plot)+1),accu_plot)
            #plt.show()
            end = time.time()
            print ('')
            print ('-'*15+'{} s'.format(end-start)+'-'*15)
            print ('*'*15+'Happy Computing'+'*'*15)
            print ('*'*17+'Quinlo Group'+'*'*17)

