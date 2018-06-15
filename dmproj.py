import os
import pandas as pd
import glob
import pickle,gzip
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt

import plots
from models import FeedForward
from sklearn.model_selection._split import train_test_split


file = '/mar/home/gsingh/DMProject/Data/2001.csv'
path = '/mar/home/gsingh/DMProject/Data'

cols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'AirTime', 
'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CarrierDelay', 'WeatherDelay', 
'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

feat = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'AirTime', 
'ArrDelay', 'DepDelay', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CarrierDelay', 'WeatherDelay', 
'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

col = ['Year', 'Month','DayofMonth', 'Origin']


def build():
    n=0
    fname = os.path.basename(path)+"DMFrames-"+str(len(cols))+"-"+"ALL"+".pkz"
    
    if os.path.isfile(fname):
            with gzip.open(fname) as f:
                return pickle.load(f)
    else:
          
        allFiles = glob.glob(path + "/*.csv")
        frame = pd.DataFrame()
        list_ = []
        for file_ in allFiles:
            df = pd.read_csv(file_,encoding = "utf-8-sig", usecols=cols,header=0)
            list_.append(df)
        frame = pd.concat(list_)
        frame.set_index("Origin", inplace=True)
        
        frame['Date'] = frame['Year'].map(str) +'-'+ frame['Month'].map(str) +'-'+ frame['DayofMonth'].map(str)
        frame.drop(frame.columns[[0,1,2]], axis=1, inplace=True)
        frame['Date']=pd.to_datetime(frame.Date.astype(str))
        targetframe = frame.loc[frame.index.isin(['OAK', 'SFO'])]
        
        with gzip.open(fname,mode='wb') as f:
                pickle.dump(targetframe,f)
        return targetframe


def driver():
    dataset = build()
    delaylist = ['ArrDelay','DepDelay','CarrierDelay', 'WeatherDelay', 
                 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    #plotStats(dataset, plotlist1, 'SFO')
    #print(dataset.columns.tolist())
    
    dataset = dataset.reset_index()
    dataset.fillna(0)
    #Converting categorical features to numerics
    dataset["Dest"] = dataset["Dest"].astype('category')
    dataset["Dest"] = dataset["Dest"].cat.codes    
    
    #dataset = dataset.sample(n=20000)
    
    dataset['Date'] = dataset['Date'].apply(lambda x: x.timestamp())
    dataSFO = dataset.loc[dataset['Origin'].isin(['SFO'])]
    dataOAK = dataset.loc[dataset['Origin'].isin(['OAK'])]
    dataSFO=dataSFO.iloc[0:10000]
    dataOAK=dataOAK.iloc[0:10000]
    frames = [dataSFO, dataOAK]
    NNdata = pd.concat(frames)
    #NNdata = NNdata.sample(n=20000)
    labels = NNdata["Origin"]
    NNdata.drop('Origin', axis=1, inplace=True)
    
    delayset = dataset[delaylist]
    
    c1=dataset.DayOfWeek.unique()
    
    
    #labels = dataset["Origin"]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = np_utils.to_categorical(labels, 2)
    data = NNdata
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8)
    
    FeedForward(x_train, x_test, y_train, y_test, len(NNdata.dtypes))
    #print(NNdata.dtypes)
    
if __name__ == '__main__':
    driver() 
