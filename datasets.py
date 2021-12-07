import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import random

def load_NBaIoT():

    # random_numbers =  [1] 
    # num = random.choice(random_numbers)
    # benign=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.benign.csv')
    # g_c=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.combo.csv')
    # g_j=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.junk.csv')
    # g_s=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.scan.csv')
    # g_t=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.tcp.csv')
    # g_u=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.udp.csv')
    # m_a=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.ack.csv')
    # m_sc=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.scan.csv')
    # m_sy=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.syn.csv')
    # m_u=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.udp.csv')
    # m_u_p=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.udpplain.csv')
    # # g_st = pd.concat([g_s,g_t])
    # # benign=benign.sample(frac=0.25,replace=False)
    # # g_c=g_c.sample(frac=0.25,replace=False)
    # # g_j=g_j.sample(frac=0.5,replace=False)
    # # g_s=g_s.sample(frac=0.15,replace=False)
    # # g_u=g_u.sample(frac=0.15,replace=False)
    # # m_a=m_a.sample(frac=0.25,replace=False)
    # # m_sc=m_sc.sample(frac=0.15,replace=False)
    # # m_sy=m_sy.sample(frac=0.25,replace=False)
    # # m_u=m_u.sample(frac=0.1,replace=False)
    # # m_u_p=m_u_p.sample(frac=0.27,replace=False)

    # benign['type']='benign'
    # m_u['type']='mirai_udp'
    # g_c['type']='gafgyt_combo'
    # g_j['type']='gafgyt_junk'
    # g_s['type']='gafgyt_scan'
    # g_u['type']='gafgyt_udp'
    # m_a['type']='mirai_ack'
    # m_sc['type']='mirai_scan'
    # m_sy['type']='mirai_syn'
    # m_u_p['type']='mirai_udpplain'

    # data=pd.concat([benign,m_u,g_c,g_j,g_s,g_u,m_a,m_sc,m_sy,m_u_p],axis=0, sort=False, ignore_index=True)

    # #how many instances of each class
    # data.groupby('type')['type'].count()

    # #shuffle rows of dataframe 
    # sampler=np.random.permutation(len(data))
    # data=data.take(sampler)
    # data.head()

    # #dummy encode labels, store separately
    # labels_full=pd.get_dummies(data['type'], prefix='type')
    # labels_full.head()

    # #drop labels from training dataset
    # data=data.drop(columns='type')
    # data.head()

    # #standardize numerical columns
    # def standardize(df,col):
    #     df[col]= (df[col]-df[col].mean())/df[col].std()

    # data_st=data.copy()
    # for i in (data_st.iloc[:,:-1].columns):
    #     standardize (data_st,i)

    # data_st.head()

    #training data for the neural net
    df_features = pd.read_csv("Kaggle/N-BaIoT/train_features.csv")
    df_labels = pd.read_csv("Kaggle/N-BaIoT/train_labels.csv")
    
    train_data_st=df_features.values
    train_data_st

    labels=df_labels.values
    labels

    x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_data_st, labels, test_size=0.25, random_state=42)
    return x_train_st, x_test_st, y_train_st, y_test_st

def load_NBaIoT_Test():
    # random_numbers =  [1] 
    # num = random.choice(random_numbers)
    # benign=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.benign.csv')
    # g_c=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.combo.csv')
    # g_j=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.junk.csv')
    # g_s=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.scan.csv')
    # # g_t=pd.read_csv('Kaggle/N-BaIoT/1.gafgyt.tcp.csv')
    # g_u=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.gafgyt.udp.csv')
    # m_a=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.ack.csv')
    # m_sc=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.scan.csv')
    # m_sy=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.syn.csv')
    # m_u=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.udp.csv')
    # m_u_p=pd.read_csv('Kaggle/N-BaIoT/'+ str(num) + '.mirai.udpplain.csv')
   
    # # g_st = pd.concat([g_s,g_t])
    # benign=benign.sample(frac=0.25,replace=False)
    # g_c=g_c.sample(frac=0.25,replace=False)
    # g_j=g_j.sample(frac=0.5,replace=False)
    # g_s=g_s.sample(frac=0.5,replace=False)
    # g_u=g_u.sample(frac=0.15,replace=False)
    # m_a=m_a.sample(frac=0.25,replace=False)
    # m_sc=m_sc.sample(frac=0.15,replace=False)
    # m_sy=m_sy.sample(frac=0.25,replace=False)
    # m_u=m_u.sample(frac=0.1,replace=False)
    # m_u_p=m_u_p.sample(frac=0.27,replace=False)

    # benign['type']='benign'
    # m_u['type']='mirai_udp'
    # g_c['type']='gafgyt_combo'
    # g_j['type']='gafgyt_junk'
    # g_s['type']='gafgyt_scan'
    # g_u['type']='gafgyt_udp'
    # m_a['type']='mirai_ack'
    # m_sc['type']='mirai_scan'
    # m_sy['type']='mirai_syn'
    # m_u_p['type']='mirai_udpplain'

    # data=pd.concat([benign,m_u,g_c,g_j,g_s,g_u,m_a,m_sc,m_sy,m_u_p],axis=0, sort=False, ignore_index=True)

    # #how many instances of each class
    # data.groupby('type')['type'].count()

    # #shuffle rows of dataframe 
    # sampler=np.random.permutation(len(data))
    # data=data.take(sampler)
    # data.head()

    # #dummy encode labels, store separately
    # labels_full=pd.get_dummies(data['type'], prefix='type')
    # labels_full.head()

    # #drop labels from training dataset
    # data=data.drop(columns='type')
    # data.head()

    # #standardize numerical columns
    # def standardize(df,col):
    #     df[col]= (df[col]-df[col].mean())/df[col].std()

    # data_st=data.copy()
    # for i in (data_st.iloc[:,:-1].columns):
    #     standardize (data_st,i)

    # data_st.head()

    #training data for the neural net
    # train_data_st=data_st.values
    # train_data_st

    # #labels for training
    # labels=labels_full.values
    # labels

    df_features = pd.read_csv("Kaggle/N-BaIoT/test_features.csv")
    df_labels = pd.read_csv("Kaggle/N-BaIoT/test_labels.csv")

    train_data_st=df_features.values
    train_data_st

    labels=df_labels.values
    labels


    # test/train split  25% test
    return train_data_st, labels

def load_UNSW():
    df_features = pd.read_csv("Kaggle/UNSW/train_features.csv")
    df_labels = pd.read_csv("Kaggle/UNSW/train_labels.csv")
    
    train_data_st=df_features.values
    train_data_st

    labels=df_labels.values
    labels

    x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_data_st, labels, test_size=0.25, random_state=42)
    return x_train_st, x_test_st, y_train_st, y_test_st

def load_UNSW_Test():
    df_features = pd.read_csv("Kaggle/UNSW/test_features.csv")
    df_labels = pd.read_csv("Kaggle/UNSW/test_labels.csv")

    train_data_st=df_features.values
    train_data_st

    labels=df_labels.values
    labels
    return train_data_st, labels


def load_BoT_IoT_Test():
    df = pd.read_csv("Kaggle/BoT-IoT/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv")
    # df_test=pd.read_csv("/content/drive/MyDrive/Datasets/Kaggle/BoT-IoT/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv")
    df=df.sample(frac=0.25,replace=False)

    #shuffle rows of dataframe 
    sampler=np.random.permutation(len(df))
    df=df.take(sampler)

    df.drop(["pkSeqID","seq","subcategory","saddr","daddr"], axis=1, inplace=True)

    indexNames = df[df['category']=='Theft'].index
    df.drop(indexNames , inplace=True)

    df['sport']=df['sport'].replace(['0x0303'],'771') 
    df['sport']=df['sport'].replace(['0x0011'],'17')
    df['sport']=df['sport'].replace(['0x000d'],'13')
    df['sport']=df['sport'].replace(['0x0008'],'8')
    df["sport"] = df["sport"].astype(str).astype(int)


    df['dport']=df.dport.apply(lambda x: int(x,16) if len(x)>1 and x[1]=="x" else int(x))
    
    le = LabelEncoder()
    df["proto_enc"]= le.fit_transform(df.proto)
    df.drop(['proto'], axis=1, inplace=True)

    #dummy encode labels, store separately
    labels_full_train=pd.get_dummies(df['category'], prefix='type')
    labels_full_train.head()

    df.drop(["category"], axis=1, inplace=True)

    scaler = Normalizer().fit(df)
    df = scaler.transform(df)

    df = pd.DataFrame(df)

    train_data_st=df.values
    train_data_st

    print(train_data_st.shape)
    #labels for training
    labels=labels_full_train.values
    labels
    print(labels.shape)
    return train_data_st, labels