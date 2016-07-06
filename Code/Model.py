import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS
from datetime import datetime
from bson.json_util import dumps
import json
from bson import json_util
import os
from pandas.io.json import json_normalize

from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import cross_validation
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from dateutil import parser
import datetime
from plotly.graph_objs import *
import boxofficemojoAPI as bom


# Get all the movies list
box_office_mojo = bom.BoxOfficeMojo()
box_office_mojo.crawl_for_urls()

# combine all the json strings

for moviename in box_office_mojo.movie_urls.iterkeys():
    movie = box_office_mojo.get_movie_summary(moviename)
    movie.clean_data()
    weekly = box_office_mojo.get_weekly_summary(moviename)
    weekly.clean_data()
    df = pd.read_json(movie.to_json())
    dfall = pd.concat([dfall, df])

def append_record(record):
    with open('my_file', 'a') as f:
        json.dump(record, f,indent=4, sort_keys=True, default=json_util.default)
        f.write(os.linesep)


for moviename in box_office_mojo.movie_urls.iterkeys():
    i += 1
    movie = box_office_mojo.get_movie_summary(moviename)
    movie.clean_data()

    weekly = box_office_mojo.get_weekly_summary(moviename)
    weekly.clean_data()
    if len(weekly.data['weekly']) >= 2:
        for k, v in movie.data.iteritems():
            if type(v) == list:
                movie.data[k] = ','.join(v)
        dict1 = movie.data.copy()
        dict2 = weekly.data['weekly'][0]
        dict1.update(dict2)
        dict1['gross2'] = weekly.data['weekly'][1]['gross']
        append_record(dict1)

# convert data to dataframe


i = 0
j = 0
x = 0

for moviename in box_office_mojo.movie_urls.iterkeys():

    movie = box_office_mojo.get_movie_summary(moviename)
    movie.clean_data()

    if (movie.data['title'] in movieall['title'].values) or (movie.data['title'] in badmovie):
        if x % 50 == 0:
            print x
        x += 1
        continue

    weekly = box_office_mojo.get_weekly_summary(moviename)
    weekly.clean_data()

    if len(weekly.data['weekly']) >= 2:
        for k, v in movie.data.iteritems():
            if type(v) == list:
                movie.data[k] = ','.join(v)
        dict1 = movie.data.copy()
        dict2 = weekly.data['weekly'][0]
        dict1.update(dict2)
        dict1['gross2'] = weekly.data['weekly'][1]['gross']

        df1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict1.iteritems()]))
        movieall = pd.concat([movieall, df1], axis=0, ignore_index=True)

        aweek = pd.DataFrame({'title': [weekly.data['title']]})
        aweek['release_date'] = [weekly.data['weekly'][0]['week']]
        aweek['length'] = [len(weekly.data['weekly'])]
        k = 1
        for weeks in weekly.data['weekly']:
            col = 'w' + str(k)
            aweek[col] = [weeks['gross']]
            k += 1
        weekall = pd.concat([weekall, aweek], axis=0, ignore_index=True)

        i += 1
        if i % 10 == 0:
            print 'got movie: ', i
    else:
        j += 1
        badmovie.append(movie.data['title'])
        if j % 10 == 0:
            print 'passed movie: ', j, '--', movie.data['title']



## slice part-done box_office_mojo.movie_urls to a shorter dict
url2 = {}

for i,moviename in box_office_mojo.movie_urls.iteritems():
    if movieall['title'].str.contains(moviename).any() :
        continue
    elif any(moviename in s for s in badmovie):
        continue
    elif i in url2.keys():
        continue
    else:
        url2[i] = moviename


#### weekall cpi adjust
weekall.fillna(0,inplace = True)
weekall['release_year'] = weekall['release_date'].map(lambda x: int(x[0:4]))

for index, row in weekall.iterrows():
    n = cpid[2016] / cpid[row['release_year']]
    weekall.ix[index,3:306] = weekall.ix[index,3:306].apply(lambda x : x*n)
    if index%100 ==0:
        print index

dfall = pd.read_csv('movieall.csv',sep='\t')
weekall = pd.read_csv('weekall.csv',sep = '\t')

#######   fill na
dfall['cinematographers'].fillna('',inplace = True)
dfall['actors'].fillna('',inplace = True)
dfall['average_per_theatre'].fillna(0,inplace = True)
dfall['composers'].fillna('',inplace = True)
dfall['directors'].fillna('',inplace = True)
dfall['distributor'].fillna('',inplace = True)
dfall['genre'].fillna('',inplace = True)
dfall['domestic'].fillna(0,inplace = True)
dfall['gross'].fillna(0,inplace = True)
dfall['gross2'].fillna(0,inplace = True)
dfall['mpaa_rating'].fillna('',inplace = True)
dfall['producers'].fillna('',inplace = True)
dfall['production_budget'].fillna(0,inplace = True)
dfall['theaters'].fillna(1,inplace = True)
dfall['runtime'].fillna(0,inplace = True)
## careful
dfall['release_year'] = dfall['release_date'].map(lambda x: int(x[0:4]))
dfall['release_year'].fillna('',inplace = True)


# select N values
# take years
# remove useless
# genre
# actors
# directors
# rating
# same week box office

# cpi adjust
cpid = pd.read_csv('cpi.csv')
cpid = dict(zip(cpi['Year'], cpi['Annual']))
for index, row in dfall.iterrows():
    dfall.set_value(index, 'domestic', row['domestic'] * cpid[2016] / cpid[row['release_year']])
    dfall.set_value(index, 'gross', row['gross'] * cpid[2016] / cpid[row['release_year']])
    dfall.set_value(index, 'gross2', row['gross2'] * cpid[2016] / cpid[row['release_year']])
    dfall.set_value(index, 'average_per_theatre', row['average_per_theatre'] * cpid[2016] / cpid[row['release_year']])

for index, row in weekall.iterrows():
    dfall.set_value(index,'domestic', row['domestic'] * cpid[2016] / cpid[row['release_year']])
    dfall.set_value(index,'gross', row['gross'] * cpid[2016] / cpid[row['release_year']])
    dfall.set_value(index,'gross2', row['gross2'] * cpid[2016] / cpid[row['release_year']])
    dfall.set_value(index,'average_per_theatre', row['average_per_theatre'] * cpid[2016] / cpid[row['release_year']])


# numbers
dfnumbers = dfall[['average_per_theatre','gross','rank','production_budget','runtime','theaters','release_date']]
dfratio = weekcpi[['ratio']]

# release how many years
dfyears =pd.DataFrame(dfall['release_year'].map(lambda x: 2016-x))

# genre
dfgenre=dfall[['genre']]
dfgenre['Action']=0
dfgenre['Drama']=0
dfgenre['Comedy']=0
dfgenre['Documentary']=0
dfgenre['Foreign']=0
dfgenre['Horror']=0
dfgenre['Sci-Fi']=0
dfgenre['Romance']=0
dfgenre['Animation']=0

for index, row in dfgenre.iterrows():
    if 'Action' in row['genre']:
        dfgenre.set_value(index, 'Action', 1)
    if 'Drama' in row['genre']:
        dfgenre.set_value(index, 'Drama', 1)
    if 'Comedy' in row['genre']:
        dfgenre.set_value(index, 'Comedy', 1)
    if 'Documentary' in row['genre']:
        dfgenre.set_value(index, 'Documentary', 1)
    if 'Foreign' in row['genre']:
        dfgenre.set_value(index, 'Foreign', 1)
    if ('Horror' in row['genre']) or ('Thriller' in row['genre']):
        dfgenre.set_value(index, 'Horror', 1)
    if 'Sci-Fi' in row['genre']:
        dfgenre.set_value(index, 'Sci-Fi', 1)
    if ('Romance' in row['genre']) or ('Romantic' in row['genre']):
        dfgenre.set_value(index, 'Romance', 1)
    if 'Animation' in row['genre']:
        dfgenre.set_value(index, 'Animation', 1)

# mpaa rating
dfmpaa = pd.get_dummies(dfall.mpaa_rating)

# actors

dfactors = dfall[['actors']]
dfactors = dfactors.fillna('')
dfactors['1'] = 0
dfactors['2'] = 0
dfactors['3'] = 0
dfactors['4'] = 0
dfactors['actors_sum'] = 0
for index, row in dfactors.iterrows():
    if len(row['actors']) == 0:
        continue
    alist = row['actors'].split(',')
    y=0
    for i in range(min(4,len(alist))):
        mask = np.column_stack([dfall['actors'].str.contains(alist[i], na=False)])
        dftemp = dfall.loc[mask.any(axis=1)]
        mask2 = np.column_stack([dftemp['release_date'] < dfall.get_value(index,'release_date')])
        x = dftemp.loc[mask2.any(axis=1)]['domestic'].sum()
        y += x
        dfactors.set_value(index,str(i+1),x)
    dfactors.set_value(index,'actors_sum', y)

# directors
dfdire = dfall[['directors']]
dfdire = dfdire.fillna('')
dfdire['dire_sum'] = 0
for index, row in dfdire.iterrows():
    if len(row['directors']) == 0:
        continue
    dlist = row['directors'].split(',')
    mask = np.column_stack([dfall['directors'].str.contains(dlist[0], na=False)])
    dftemp = dfall.loc[mask.any(axis=1)]
    mask2 = np.column_stack([dftemp['release_date'] < dfall.get_value(index, 'release_date')])
    x = dftemp.loc[mask2.any(axis=1)]['domestic'].sum()
    dfdire.set_value(index, 'dire_sum', x)

# distributor
dfdist = dfall[['distributor']]
#dfdist = dfdist.fillna('')
dfdist['dist_sum'] = 0
for index, row in dfdist.iterrows():
    if len(row['distributor']) == 0:
        continue
    dlist = row['distributor'].split(',')
    mask = np.column_stack([dfall['distributor'].str.contains(dlist[0], na=False)])
    dftemp = dfall.loc[mask.any(axis=1)]
    mask2 = np.column_stack([dftemp['release_date'] < dfall.get_value(index,'release_date')])
    x = dftemp.loc[mask2.any(axis=1)]['domestic'].sum()
    dfdist.set_value(index,'dist_sum',x)

##### build X, y

X = pd.concat([dfnumbers,dfratio,dfyears,dfactors,dfdire,dfdist],axis =1)
y = dfall[['gross2']]
X.fillna(0,inplace=True)
y.fillna(0,inplace=True)
X.to_csv('X.csv', sep='\t',index = False)
y.to_csv('y.csv', sep='\t',index = False)

# ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = linear_model.RidgeCV(alphas=[40, 30, 35])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print clf.score(X_test,y_test),clf.alpha_


def rerank(x):
    return x*500
#X['rank'] = X['rank'].apply(rerank)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
params = {'max_depth': 5, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

data = [go.Bar(x=X_train.columns[sorted_idx],
               y=feature_importance[sorted_idx])]
py.iplot(data, filename='Feature Importance')

# clusters

X = pd.concat([dfnumbers,dfratio,dfgenre,dfactors,dfdire,dfyears],axis =1)
X['theaters'].fillna(1,inplace=True)
X.fillna(0,inplace=True)
y = dfall[['gross2']]

Xy = pd.concat([X,y],axis = 1)
X = Xy[Xy['Action'] == 1 ]
#X = X[X['average_per_theatre']<10000]
y=X[['gross2']]
X = X[['average_per_theatre','gross','production_budget','runtime','theaters','ratio','actors_sum','dire_sum','release_year']]

# bar
#py.sign_in('username', 'api_key')
xaxis = np.linspace(0,len(y_test)-1,len(y_test))
trace1 = Bar(
    x=xaxis,
    y=y_pred,
    name='Predict Value',
    uid='8e60d4'
)
trace2 = Bar(
    x=xaxis,
    y=y_test.values,
    name='True Value',
    uid='4ab239'
)
data = Data([trace1, trace2])
layout = Layout(
    autosize=True,
    height=711,
    legend=Legend(
        font=Font(
            size=12
        )
    ),
    showlegend=True,
    title='Predict Value vs True Value',
    width=1100,
    xaxis=XAxis(
        autorange=True,
        type='category'
    ),
    yaxis=YAxis(
        autorange=True,
        type='linear'
    )
)
fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='Prediction')



