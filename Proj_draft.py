
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import os
import re
import numpy as np 
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from numpy import array
from numpy import argmax
from numpy import array_equal
from sklearn.svm import SVR
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.metrics import mean_squared_error


global filename,filename1
global model, encoder_model, decoder_model
global dataset
global lstm_error,elm_error
list = []

def upload():
    global filename
    list.clear()
    filename = open("C:/Users/Krish/Desktop/maj_proj/datasetss.txt","r")
    for line in filename:
        line = line.strip('\n')
        list.append(line)
    print("\nDataset is uploaded\n")
                        

def lstmModel():
    global model, encoder_model, decoder_model
    # define training encoder
    encoder_inputs = Input(shape=(None, 9))
    encoder = LSTM(512, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, 9))
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)   #LSTM with SEQ2SEQ object sequences created here
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	
    decoder_dense = Dense(9, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(512,))
    decoder_state_input_c = Input(shape=(512,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    print("LSTM Model Generated\n")

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
    
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = []
	for t in range(n_steps):
		# predict next char
		teacher_ratio, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(teacher_ratio[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = teacher_ratio
	return array(output) 

def trainLSTM():
    
    global dataset
    global model
    global lstm_error
    filename1=open("C:/Users/Krish/Desktop/maj_proj/data1.csv","r")
    train = pd.read_csv(filename1)
    size = len(train)
    dataset = np.zeros((size, 9, 9))

    m = 0;
    n = 0
    p = 0

    for i in range(len(train)) :
        person = int(train.iloc[i, 0])
        position = int(train.iloc[i, 1])
        latitude = float(train.iloc[i, 2])
        longitude = float(train.iloc[i, 3])
        n = 0
        for j in range(len(train)):
            person1 = int(train.iloc[j, 0])
            position1 = int(train.iloc[j, 1])
            latitude1 = float(train.iloc[j, 2])
            longitude1 = float(train.iloc[j, 3])
            if person == person1:
                dataset[m][position1-1][n] = latitude1
                n = n + 1
                dataset[m][position1-1][n] = longitude1
                n = n + 1
                dataset[m][position1-1][n] = person
                n = n + 1
                if n >= 9:
                    n = 0
                    
        m = m + 1
    print(dataset.shape)    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    # summarize defined model
    print(model.summary())
    model.fit([dataset,dataset], dataset, epochs=10)
    scores = model.evaluate([dataset,dataset], dataset, verbose=2)
    accuracy = scores[1]*100
    lstm_error = 100.0 - accuracy
    
    
    print("LSTM MSE Error : "+str(lstm_error)+"\n")
    model.save("draft_traj_model.h5py")
    

def predict():
    latitude = input("Enter Current Latitude Location Value")
    longitude = input("Enter Current Longitude Location Value")
    user = input("Enter User ID")
    b = np.zeros((1, 9, 9))
    b[0][0][0] = float(latitude)
    b[0][1][1] = float(longitude)
    b[0][2][2] = int(user)
    lat = ''
    lon = ''
    for i in range(len(list)):
        if i > 0:
            arr = list[i].split(",")
            if float(latitude) == float(arr[2]) and float(longitude) == float(arr[3]):
                arr = list[i+1].split(",")
                lat = arr[2]
                lon = arr[3]
                break
            
    target = predict_sequence(encoder_model, decoder_model, b, 3, 9)
    output = one_hot_decode(target)
    print("Predicted Sequences of users next steps are : \n\n")
    #print(output)
    #print(str(output[0])+" "+str(output[1])+" "+str(output[2]))
    print("Next Location Latiitude : "+lat+"\n\n")
    print("Next Location Longitude : "+lon+"\n\n")
    #print("Next Sequences : "+str(dataset[int(user)][output[0]])+"\n")
    #print("Next Sequences : "+str(dataset[int(user)][output[1]])+"\n")
    #print("Next Sequences : "+str(dataset[int(user)][output[2]])+"\n")
   

def extension():
    global elm_error
    filename1=open("C:/Users/Krish/Desktop/maj_proj/data1.csv","r")
    user = input("Enter User ID")
    print("\n")
    train = pd.read_csv(filename1)
    X = train.values[:, 0:4] 
    y = train.values[:, 0]

    trainX = []
    trainY = []
    trainY1 = []
    for i in range(len(X)):
        usr = X[i][0]
        x_loc = X[i][2]
        y_loc = X[i][3]
        if str(usr) == user:
            trainY.append(x_loc)
            trainY1.append(y_loc)
            trainX.append([x_loc,y_loc])
            
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)
    trainY1 = np.asarray(trainY1)
    print(trainX)
    
    #svm_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)    
    #svm_rbf.fit(trainX, trainY)    
    #print(svm_rbf.predict(trainX))

    svm_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)    
    #svm_rbf.fit(trainX, trainY1)    
    #print(svm_rbf.predict(trainX))
    
    srhl_tanh = MLPRandomLayer(n_hidden=200, activation_func='tanh')
    cls = ELMRegressor(regressor=svm_rbf)
    cls.fit(trainX, trainY, epochs=10)
    y_pred = cls.predict(trainX)

    srhl_tanh = MLPRandomLayer(n_hidden=200, activation_func='tanh')
    cls = ELMRegressor(regressor=svm_rbf)
    cls.fit(trainX, trainY1)
    y_pred1 = cls.predict(trainX)

    err = []
    for i in range(len(y_pred)):
        err.append([y_pred[i],y_pred1[i]])
    err = np.asarray(err)    
    elm_error = mean_squared_error(trainX, err)

    length = len(y_pred) - 1
    print("\nELM Extension Next Predicted Sequence is :\n")
    print("Latitude : "+str(y_pred[length])+"\n")
    print("Longitude : "+str(y_pred1[length])+"\n")
    print("ELM MSE Error : "+str(elm_error))
    
def graph():
    height = [lstm_error,elm_error]
    bars = ('LSTM MSE Error','ELM MSE Error')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
def main():
    upload()
    lstmModel()
    trainLSTM()
    predict()
    extension()
    graph()

if __name__=="__main__":
    main()