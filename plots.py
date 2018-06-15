import matplotlib.pyplot as plt
import pandas as pd    

def plotDayOfWeek(dataset):    
    # Set title
    ttl = 'Weekly Flight Frequencies'
    # Create a colormap
    cmap=plt.cm.get_cmap('Blues')
    ax=dataset['DayOfWeek'].value_counts().sort_index().plot(kind='pie', figsize=(6,6),
                               colormap=cmap,title=ttl)
    ax.legend(bbox_to_anchor=(0.9, 1.1))
    plt.show()
'''    
def plotDelays(dataset):
    
    ttl = 'Weekly Flight Frequencies'
    # Create a colormap
    cmap=plt.cm.get_cmap('Blues')
    ax=dataset['DayOfWeek'].value_counts().sort_index().plot(kind='pie', figsize=(6,6),
                               colormap=cmap,title=ttl)
    ax.legend(bbox_to_anchor=(0.9, 1.1))
    plt.show()
'''   
    
def plotAllDelaysBar(delayset):
    delaylist = ['ArrDelay','DepDelay','CarrierDelay', 'WeatherDelay', 
                 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    
    fig, axes = plt.subplots(nrows=len(delaylist), ncols=1)
    for i, c in enumerate(delayset.columns):
        delayset[c].value_counts().sort_index().plot(kind='bar', ax=axes[i], figsize=(12, 10), title=c)
    plt.show()
    
def plotAllDelaysPie(delayset):
    cmap=plt.cm.get_cmap('hot')
    ax=delayset.count().plot(kind='pie', figsize=(10,10),labels=None,
                   title='Delay Contributions', colormap=cmap)
    ax.legend(bbox_to_anchor=(0.9, 1.1), labels=delayset[0:])
    ax.set_ylabel('')
    plt.show()
    
def plotYearlyFlights(dataset):
    ttl = 'Yearly Flights out of '
    dataSFO = dataset.loc[dataset['Origin'].isin(['SFO'])]
    dataOAK = dataset.loc[dataset['Origin'].isin(['OAK'])]
    fig, axes = plt.subplots(nrows=len(dataset.Origin.unique()), ncols=1)
    
    cmap=plt.cm.get_cmap('jet')
    a1=dataSFO.Year.value_counts().sort_index().plot(kind='bar', ax=axes[0],
                             figsize=(12, 10),colormap=cmap, title=ttl+'SFO')
    a2=dataOAK.Year.value_counts().sort_index().plot(kind='bar', ax=axes[1],
                             figsize=(12, 10),colormap=cmap, title=ttl+'OAK')
    
    a1.set_xticklabels([])
    a1.set_ylabel('No. of Departure Flights')
    a2.set_xlabel('Year')
    plt.show()
    
def plotCancellations(dataset):
    cdata=dataset.loc[dataset.Cancelled.isin([1])]
    ttl = 'Cancelled Flights per Year'
    cmap = plt.cm.get_cmap('viridis')
    ax=cdata.Year.value_counts().sort_index().plot(kind='barh',
                             figsize=(12, 10),colormap=cmap, title=ttl)
    ax.set_xlabel('No. of Cancellations')
    ax.set_ylabel('Year')
    plt.show()