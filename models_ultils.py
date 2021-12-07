#import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,Activation
from keras.layers import LSTM, SimpleRNN, GRU

def load_ann(input_dimension, output_dimension):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dimension, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30, input_dim=input_dimension, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, input_dim=input_dimension, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(output_dimension,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics="accuracy")
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics='categorical_accuracy')
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics="accuracy")
    return model

def load_ann2(input_dimension, output_dimension):

    # 1. define the network
    model = Sequential()
    model.add(Dense(1024,input_dim=input_dimension,activation='relu'))  
    model.add(Dropout(0.01))
    model.add(Dense(768,activation='relu'))  
    model.add(Dropout(0.01))
    model.add(Dense(512,activation='relu'))  
    model.add(Dropout(0.01))
    model.add(Dense(256,activation='relu'))  
    model.add(Dropout(0.01))
    model.add(Dense(output_dimension))
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def load_gru(input_dimension, output_dimension):
    # 1. define the network
    model = Sequential()
    model.add(GRU(64,input_dim=input_dimension, return_sequences=True))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(GRU(64,return_sequences=True))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(GRU(64, return_sequences=True))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(GRU(64, return_sequences=False))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(Dense(output_dimension))
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def load_lstm(input_dimension, output_dimension):
    # 1. define the network
    model = Sequential()
    model.add(LSTM(64,input_dim=41, return_sequences=True))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(LSTM(64,return_sequences=True))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=True))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=False))  # try using a GRU instead, for fun
    model.add(Dropout(0.1))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model