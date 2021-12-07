import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
#import libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
from tensorflow.keras import models


df_features = pd.read_csv("../Kaggle/N-BaIoT/train_features.csv")
df_labels = pd.read_csv("../Kaggle/N-BaIoT/train_labels.csv")

train_data_st=df_features.values
train_data_st

labels=df_labels.values
labels

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_data_st, labels, test_size=0.25, random_state=42)


# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=x_train_st.shape[1],activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(y_train_st.shape[1]))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


history = model.fit(x_train_st,y_train_st,validation_data=(x_test_st, y_test_st),epochs=50)
models.save_model(model, './NBaIoT-ANN-50e-712.h5')

print(history.history)
hist_df = pd.DataFrame(history.history) 
# or save to csv: 
hist_csv_file = './NBaIoT.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

