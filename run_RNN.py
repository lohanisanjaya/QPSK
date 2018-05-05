import main_RNN as RNN
import sys
local_rnn_lstm = RNN.Local_Rnn_LSTM(eta = 0.015, epochs = int(sys.argv[1]), \
      batch_size = 400,  
      time_steps = int(sys.argv[2]),
      num_inputs = 1, 
      hidden_units = 80, 
      num_classes = 4) #len(x_data)/num_inputs minus time-steps

local_rnn_lstm.Run_LSTM_DNN(keys_size=4,Network = 'DNN_LSTM')
