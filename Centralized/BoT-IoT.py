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

df = pd.read_csv("../Kaggle/BoT-IoT/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv")
df_test=pd.read_csv("../Kaggle/BoT-IoT/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv")

# df=df.sample(frac=0.1,replace=False)

history_loss = []
history_acc = []

#shuffle rows of dataframe 
sampler=np.random.permutation(len(df))
df=df.take(sampler)
df.head()

#shuffle rows of dataframe 
sampler=np.random.permutation(len(df_test))
df_test=df_test.take(sampler)
df_test.head()

df_test.drop(["pkSeqID","seq","subcategory","saddr","daddr"], axis=1, inplace=True)
df.drop(["pkSeqID","seq","subcategory","saddr","daddr"], axis=1, inplace=True)

indexNames = df[df['category']=='Theft'].index
df.drop(indexNames , inplace=True)

indexNames = df_test[df_test['category']=='Theft'].index
df_test.drop(indexNames , inplace=True)

df['sport']=df['sport'].replace(['0x0303'],'771') 
df['sport']=df['sport'].replace(['0x0011'],'17')
df['sport']=df['sport'].replace(['0x000d'],'13')
df['sport']=df['sport'].replace(['0x0008'],'8')
df["sport"] = df["sport"].astype(str).astype(int)

df_test['sport']=df_test['sport'].replace(['0x0303'],'771') 
df_test['sport']=df_test['sport'].replace(['0x0011'],'17')
df_test['sport']=df_test['sport'].replace(['0x000d'],'13')
df_test['sport']=df_test['sport'].replace(['0x0008'],'8')
df_test["sport"] = df_test["sport"].astype(str).astype(int)

df['dport']=df.dport.apply(lambda x: int(x,16) if len(x)>1 and x[1]=="x" else int(x))
df_test['dport']=df_test.dport.apply(lambda x: int(x,16) if len(x)>1 and x[1]=="x" else int(x))

le = LabelEncoder()
df["proto_enc"]= le.fit_transform(df.proto)
df_test["proto_enc"]= le.fit_transform(df_test.proto)

df.drop(['proto'], axis=1, inplace=True)
df_test.drop(['proto'], axis=1, inplace=True)

#dummy encode labels, store separately
y_train=pd.get_dummies(df['category'], prefix='type')

y_test =pd.get_dummies(df_test['category'], prefix='type')

df_test.drop(["category"], axis=1, inplace=True)
df.drop(["category"], axis=1, inplace=True)

scaler = Normalizer().fit(df)
df = scaler.transform(df)


scaler = Normalizer().fit(df_test)
df_test = scaler.transform(df_test)

df = pd.DataFrame(df)
df_test = pd.DataFrame(df_test)

x_train=df.values
x_test=df_test.values

y_train=y_train.values
y_test=y_test.values

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    x_train, y_train, test_size=0.25, random_state=42)

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
models.save_model(model, './BoTIoT-ANN-50e-612.h5')

print(history.history)
hist_df = pd.DataFrame(history.history) 
# or save to csv: 
hist_csv_file = './BoT.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

