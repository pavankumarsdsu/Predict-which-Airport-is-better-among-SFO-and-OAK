import pandas as pd
import glob
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def model_accuracy(trained_model, features, targets):
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score


def preprocess_data(frame):
    frame['Delay'] = frame['ArrDelay'] + frame['DepDelay'] + frame['CarrierDelay'] + frame['WeatherDelay'] + frame['NASDelay'] + frame['SecurityDelay']+ frame['LateAircraftDelay']
    frame = frame[features_after_scaling]

    # handle missing values
    frame = Imputer().fit_transform(frame)
    return frame


def scale_data(data):
    scalar = preprocessing.RobustScaler()
    data = scalar.fit_transform(data)
    return data

def draw_corelated(corr):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

path = "C:/Users\Dell Pc\Documents\data-mining\data-mining";

cols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'AirTime',
        'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CarrierDelay',
        'WeatherDelay',
        'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

features = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'AirTime',
            'ArrDelay', 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CarrierDelay', 'WeatherDelay',
            'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

features_after_scaling = ['Month','DepTime', 'ArrTime', 'ActualElapsedTime', 'AirTime', 'Delay', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled']

final_features = ['Month','DepTime', 'ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn']

ListOfClassifiers = [
	LogisticRegression(),
    GaussianNB(),
	RandomForestClassifier(max_depth=50),
	AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=18, min_samples_leaf=25, min_samples_split=10),
					   n_estimators=10)
]

# load all csv
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, encoding="ISO-8859-1", usecols=cols, nrows=300000, header=0)
    list_.append(df)
frame = pd.concat(list_)

# filter SFO and OAK
is_sfo_or_oak = (frame['Origin'] == 'SFO') | (frame['Origin'] == 'OAK')
frame = frame[is_sfo_or_oak]
is_sfo = (frame['Origin'] == 'SFO')
is_oak = (frame['Origin'] == 'OAK')

sfo = frame[is_sfo]
oak = frame[is_oak]

# encode output values
frame['Origin'] = frame['Origin'].apply(lambda x: 0 if x == 'SFO' else (1 if x == 'OAK' else 2))
frame = frame.reset_index();

print("Total Count of Data Sets", frame['Origin'].value_counts())

frame_without_dest = frame[features]
frame_without_dest = preprocess_data(frame_without_dest)

df = pd.DataFrame(data=frame_without_dest,columns=features_after_scaling)
print(df)

df['label'] = df['Delay']+df['TaxiOut']
threshold = df['label'].mean()/2;
df['label'] = df['label'].apply(lambda x: 1 if x <= threshold else 0)

draw_corelated(df.corr())

# get features using final features
X = scale_data(df[final_features])
y = df['label']

train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7)

for classifier in ListOfClassifiers:
	trained_model = classifier.fit(train_x, train_y)
	test_model = classifier.fit(test_x, test_y)
	train_accuracy = model_accuracy(trained_model, train_x, train_y)
	test_accuracy = model_accuracy(trained_model, test_x, test_y)
	print("---------------",classifier,"----------------------")
	print( "Train Accuracy :: ", train_accuracy)
	print( "Test Accuracy  :: ", test_accuracy)