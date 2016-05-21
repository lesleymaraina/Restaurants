
from ingest.ingest import IngestSystem
import roperations
import pymongo
import os
import csv
import sqlite3
import nltk
import re
from pprint import pprint
from nltk.corpus import stopwords
import pickle
#from nlp.proc_lang import process_chunks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import math
import json

#nltk.download('punkt')
#nltk.download('stopwords')

"""
Use this section after making changes to the Restaurant object and ingest program and re-loading MongoDB
"""


################################################################################
#
# cities = [{'state': 'dc', 'city': 'washington'}, {'state': 'ny', 'city': 'new-york'}, {'state': 'ca', 'city': 'san-francisco'},
#             {'state': 'pa', 'city': 'philadelphia'}, {'state': 'tx', 'city': 'austin'}, {'state': 'nc', 'city': 'charlotte'},
#             {'state': 'il', 'city': 'chicago'}, {'state': 'ga', 'city': 'atlanta'}, {'state': 'wi', 'city': 'milwaukee'}, {'state': 'ca', 'city': 'los-angeles'}]
#
#
# cities = [{'state': 'dc', 'city': 'washington'}]
# loader = IngestSystem(cities)
# loader.pull_and_load()
#
# for i in loader.final_rlist: print(i['name'])
# try:
#     with open('loaderfinalrlist.pickle', 'wb') as handle:
#         pickle.dump(loader.final_rlist, handle)
#         print("Saved final list!")
# except:
#     print("Could not save final list")
################################################################################

## LOAD USER INPUTS FROM FLASK ##
yn = 'n'
while yn != 'y': yn = input("Have you entered your inputs? (y/n): ")
print("Loading user inputs...")
print()
with open('user_inputs.pickle', 'rb') as handle: u = pickle.load(handle)

userlist = [{}]
print("Keywords: ",u[0]); userlist[0]['wrd_list'] = u[0]
print("Latitude: ",u[1]); userlist[0]['latitude'] = u[1]
print("Longitude: ",u[2]); userlist[0]['longitude'] = u[2]
print("Category: ",u[3]); userlist[0]['type_2'] = u[3]
print("Price Level: ",u[4]); userlist[0]['price_level'] = u[4]
print("Rating Level: ",u[5]); userlist[0]['rating_level'] = u[5]
print("Metro Line: ",u[6]); userlist[0]['metro'] = u[6]

print()

## LOAD LIST OF ALL RESTAURANT DICTIONARIES AND TRANSFORM/ADD FIELDS ##
with open('loaderfinalrlist.pickle', 'rb') as handle:
    l = roperations.mod_r(pickle.load(handle))

## DUMP MODIFIED RESTAURANT DICTIONARY LIST ##
with open('dictlist.pickle', 'wb') as handle:
    pickle.dump(l, handle)

## JBB - For final version, only need to load dictlist.pickle and not do additional transformations

## JBB - replace pipeline code with loading pickle objects

#roperations.print_metros_dc(l) ## PRINT NEAREST RESTAURANTS IN DC AND THEIR METRO STATION ##

roperations.export_restaurants(l) ## EXPORT RESTAURANT INFO AND MENUS ##

## DEFINE FEATURE TRANSFORMATION PIPELINES ##
class ValueByKey(BaseEstimator, TransformerMixin):
    """
    Extracts a value from a dictionary by key.
    """
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, dicts):
        """Returns a list of values by key. """
        return [d[self.key] for d in dicts ]

features = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('cuisinetype', Pipeline([
                ('vbk', ValueByKey('type_2')),
                ('labels', preprocessing.LabelBinarizer()),
            ])),
            ('nlp', Pipeline([
                ('vbk', ValueByKey('wrd_list')),
                ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english'))
                #('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english'))
            ]))
        ]
    ))
])

just_txt = Pipeline([
    ('vbk', ValueByKey('wrd_list')),
    ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english')),
])

## THIS CODE WILL CREATE TRANSFORMED PICKLE OBJECTS WHICH CAN BE LOADED INTO DATA ##
# print("Creating transformations......")
# roperations.create_store_transforms(l)
# print("Done creating transformations.")

print()
if userlist[0]['type_2'] == 'Suprise me!':
    print("Loading transformed text.")
    data=just_txt.fit_transform(l)
#    with open('transforms/just_txt.pickle', 'rb') as handle:
#        data = pickle.load(handle)
else:
    print("Loading transformed categories and text.")
    data=features.fit_transform(l)
#    with open('transforms/txt_cat.pickle', 'rb') as handle:
#        data = pickle.load(handle)
print()
print("Done loading.")
print()


## FIT CLUSTERING MODEL ##
from sklearn.cluster import KMeans, MiniBatchKMeans
k=12
km = KMeans(n_clusters=k, init='k-means++')
print("Clustering sparse data with %s" % km)

labels = km.fit_predict(data)

clusters = {}
n = 0
for item in labels:
    if item in clusters: clusters[item].append(l[n]['name'])
    else: clusters[item] = [l[n]['name']]
    n +=1

## PRINT CLUSTERS ##
for item in clusters:
    print("Cluster", item)
    for i in clusters[item]: print(i)

data = {'name': [], 'cluster': [], 'latitude': [], 'longitude': [], 'rating_level': [], 'price_level': [], 'type_2': [], 'city': []}
n = 0
for item in labels:
    data['name'].append(l[n]['name'])
    data['rating_level'].append(l[n]['rating_level'])
    data['price_level'].append(l[n]['price_level'])
    data['type_2'].append(l[n]['type_2'])
    data['latitude'].append(l[n]['latitude'])
    data['longitude'].append(l[n]['longitude'])
    data['city'].append(l[n]['city'])
    #data['metros'].append(l[n]['all_mline'])
    data['cluster'].append(int(item))
    n +=1

df = pd.DataFrame(data)
df['dist'] = ''

if userlist[0]['type_2'] == 'Suprise me!':
    data=just_txt.transform(userlist)
else:
    data=features.transform(userlist)

new_lbl = km.predict(data)
print()
print("Matched cluster: ", new_lbl)
print()
print(type(new_lbl[0].item()))


## DISPLAY RESULTS ##
for i in range(0,len(df.name)):
    df.set_value(i,'dist',math.sqrt(pow(df.latitude[i].item() - userlist[0]['latitude'],2) + pow(df.longitude[i].item() - userlist[0]['longitude'],2)))

results = df[df.cluster == new_lbl[0].item()].sort_values('dist', ascending=True)
results = results[results.city == results['city'].iloc[0]]
print(results)
print(type(results))

#if userlist[0]['price_level'] != 'No preference.':
#    results = results[results.price_level == userlist[0]['price_level']]
#if userlist[0]['rating_level'] != 'No preference.':
#    results = results[results.rating_level == userlist[0]['rating_level']]
#if userlist[0]['metro'] != 'No preference.':
#    results = results[results.metros.find(userlist[0]['metro'])]

#print(results)


## Get final results and output to Excel, which is live connected to Tableau
writer = pd.ExcelWriter('output_test.xlsx')
results.to_excel(writer,'Sheet1')
writer.save()

#print(df[df.cluster == new_lbl[0].item() and df.price_level.tolower() == nri['price_level'].tolower()].sort('dist', ascending=True).head())
#print(df[df.cluster == new_lbl[0].item()].sort('dist', ascending=True).head())







##### MongoDB Export
# conn=pymongo.MongoClient()
# print(conn.database_names())
# db=conn.rdata
# print(db.collection_names())
#
# os.system("mongoexport --db rdata --collection restaurants --type=csv --fields name,street,city,city_group,state,zip,latitude,longitude,type,type_2,avg_rating --out /Users/jbbinder/Desktop/new_output.csv")
#
# db.restaurants.aggregate([
#     {'$unwind': '$menu'},
#     {'$project': {'_id': 0, 'name': 1, 'street': 1, 'city': 1, 'city_group': 1, 'state': 1, 'zip': 1, 'latitude': 1, 'longitude': 1, 'cuisinetype': '$type', 'cuisinetype_2': '$type_2', 'avg_rating': 1, 'item': '$menu.item', 'price': '$menu.price', 'description': '$menu.description'}},
#     {'$out': 'aggregate_restaurants'}
# ])
#
# print(db.collection_names())
#
# os.system("mongoexport --db rdata --collection aggregate_restaurants --type=csv --fields name,street,city,city_group,state,zip,latitude,longitude,cuisinetype,cuisinetype_2,avg_rating,item,price,description --out /Users/jbbinder/Desktop/new_output_allmenuitems.csv")




##### Load data into sqlite3
# DBPATH = '/Users/jbbinder/Desktop/sql_rdb.db'
# conn = sqlite3.connect(DBPATH)
# cur = conn.cursor()
#
# cur.execute("CREATE TABLE r (name,street,city,city_group,state,zip,latitude,longitude,cuisinetype,cuisinetype_2,avg_rating REAL,item,price INTEGER,description);")
#
# with open('/Users/jbbinder/Desktop/new_output_allmenuitems.csv','r') as fin:
#    dr = csv.DictReader(fin) # comma is default delimiter
#    to_db = [(i['name'], i['street'], i['city'], i['city_group'], i['state'], i['zip'], i['latitude'], i['longitude'], i['cuisinetype'], i['cuisinetype_2'], i['avg_rating'], i['item'], i['price'], i['description']) for i in dr]
#
# cur.executemany("INSERT INTO r (name,street,city,city_group,state,zip,latitude,longitude,cuisinetype,cuisinetype_2,avg_rating,item,price,description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
# conn.commit()
#
# cur.execute("select name, item, price from r where description like '%burger%' and price < 8")
# print(cur.fetchall())
