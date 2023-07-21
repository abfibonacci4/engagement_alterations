import threading
import matplotlib.pyplot as plt
import os, time
import shutil
import numpy as np
import random
import tensorflow.compat.v1 as tf
import copy
import time
from pynput.keyboard import Key, Listener
import pandas as pd
import csv
import os

import md_config as cfg
from feature_collection import FeatureCollection


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed, GlobalAveragePooling1D, Activation, Concatenate, \
	InputLayer, PReLU
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

interval_duration = 10.0
#counter = [0]

def define_model(hparams, model_name):
	current_n_lstms = hparams['NUM_LSTM_LAYERS']
	current_lstm_units = hparams['LSTM_UNITS']
	current_n_denses = hparams['NUM_DENSE_LAYERS']
	current_dense_units = hparams['DENSE_UNITS']
	current_dropout_rates = hparams['DROPOUT_RATES']
	current_time_step = hparams['TIME_STEP']
	current_input_units = hparams['INPUT_UNITS']
	current_densen_act = hparams['ACTIVATION_F']

	model = Sequential()
	if hparams['FC1'][1] > 0:
		model.add(TimeDistributed(Dense(hparams['FC1'][1], activation='relu'),
								  input_shape=(current_time_step, hparams['FC1'][0])))

	model.add(
		CuDNNLSTM(current_lstm_units[0], return_sequences=True, input_shape=(current_time_step, current_input_units),
				  stateful=False))

	if current_n_lstms > 1:
		for idx in range(1, current_n_lstms):
			model.add(CuDNNLSTM(current_lstm_units[idx], return_sequences=True))

	for idx in range(current_n_denses):
		model.add(TimeDistributed(Dense(current_dense_units[idx], activation='relu')))

	model.add(TimeDistributed(Dense(1, activation=current_densen_act)))
	model.add(GlobalAveragePooling1D())

	return model

def get_model(model_index, n_segments=15, input_units=60):
    """
    Make prediction for data_npy
    :param data_npy:
    :return:
    """
    ld_cfg = cfg.md_cfg
    hparams = copy.deepcopy(ld_cfg[model_index])
    ft_type = 'of'


    hparams['TIME_STEP'] = n_segments
    hparams['INPUT_UNITS'] = hparams['FC1'][1] if hparams['FC1'][1] > 0 else input_units
    hparams['optimizer'] = 'adam'
    hparams['ACTIVATION_F'] = 'tanh'
    hparams['CLSW'] = 1

    cur_model = define_model(hparams,hparams['NAME'])
    cur_model.build()
    cur_model.load_weights(
            './models/{}_{}_models_{}_{}_0_epochs{}_best_weight.h5'.format(hparams['model_path'], ft_type,
                                                                           hparams['n_segments'], hparams['alpha'],
                                                                           hparams['EPOCHS']))

    return cur_model

#add path parameter
def periodic_function():
	print("processing starting")
	#print("inside periodic function")
	duration = time.strftime("%M:%S", time.gmtime(int(time.time() - start_time)))
	if os.path.isdir("OpenFace/build/myprocessed"):
		#with Listener(on_press = show) as listener:
				#listener.join()
		#print("just entered processed!")
		#add for loop to go through csv files in directory
		df = pd.read_csv(r'OpenFace/build/myprocessed/data1.csv', header=0, sep=',')
		maximum = df.shape[0]
		window_length = 20
		upper_lim = maximum - window_length
		for i in range(0, upper_lim, window_length):
			print("window " + str(i))
			feature_extraction = FeatureCollection('OpenFace/build/myprocessed', i, i + window_length)
			ft = np.array(feature_extraction.get_all_data())
			#print("just got data")

			with session1.as_default():
				with graph1.as_default():
				    v1 = eye_gaze_v1.predict(ft[0].reshape(1,15,60))
				    #print("set v1")

			with session2.as_default():
				with graph2.as_default():		
				    v2 = eye_gaze_v2.predict(ft[0].reshape(1,15,60))
				    #print("set v2")



			#print('{} {}'.format(v1,v2))
			enga_score = 0.5 * (v1 + v2)
			#print('engagement_score = {}'.format(enga_score))
			x.append(i)
			if enga_score < 0.4:
				y.append(0)
			elif enga_score < 0.6:
				y.append(1)
			elif enga_score < 0.83:
				y.append(2)
			else:
				y.append(3)
			#print("x: " + str(x))
			#print("y: " + str(y))
			i += 1
			
		#shutil.rmtree('OpenFace/build/myprocessed', ignore_errors=False)
	else:
		print("did not find processed dir")

def startTimer():
	#threading.Timer(interval_duration,startTimer).start()

	periodic_function()

if __name__ == '__main__':
	x = []
	y = []

	graph1 = tf.Graph()
	with graph1.as_default():
		session1 = tf.Session()
		with session1.as_default():
			eye_gaze_v1 = get_model(model_index=0)
	graph2 = tf.Graph()
	with graph2.as_default():
		session2 = tf.Session()
		with session2.as_default():
			eye_gaze_v2 = get_model(model_index=1)

	start_time = time.time()
	startTimer()
	print("ending program")
	path = r'OpenFace/build/myprocessed/results/scores.csv'
	with open(path, 'a') as csvfile:
		csvwriter = csv.writer(csvfile)
		#go through videodata csv files in directory, write to scores csv
		#also add a row for a header to indicate what video the data corresponds to
		df = pd.read_csv(r'OpenFace/build/myprocessed/data1.csv', header=0, sep=',')
		maximum = df.shape[0]
		window_length = 20
		upper_lim = maximum - window_length
		for j in range(0, upper_lim):
			csvwriter.writerow([x[j], y[j]])
		print("saved to csv")		
		exit()
	while True:
		plt.yticks(np.arange(4), ('Disengaged', 'Barely Engaged', 'Engaged', 'Highly Engaged'))
		plt.xticks(rotation=90)
		plt.step(x, y, 'b')
		plt.pause(1)
		
		
