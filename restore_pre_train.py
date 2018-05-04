import tensorflow as tf
import data_loader as dl
import numpy as np


sess = tf.Session()

saver = tf.train.import_meta_graph('pre_trained_net.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()

Accuracy = graph.get_tensor_by_name('Accuracy:0')
prediction_series = graph.get_tensor_by_name('predition_series:0')
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
accu_plot = []
for i in np.linspace(200,2000,10):
    print ('upto', i)
    testX,testY = dl.data_reader_test_portion(filename='keys_1_test.pkl',upto=int(i))
    test_X_in = testX.reshape(200,2000,1)
    test_Y_in = testY.reshape(200,2)
    accu = sess.run(Accuracy,feed_dict={X:test_X_in,Y:test_Y_in})
    print ('accuracy',accu)
    accu_plot.append(accu)
print (accu_plot)
    #print ('prediction_series',pred)
