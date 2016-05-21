from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

with open('loaderfinalrlist.pickle', 'rb') as handle:
    l = pickle.load(handle)
token_dict = {}
for i in l:
    tmp = ""
    max_pr = 0
    for w in i['menu']:
        if w['item']: tmp=tmp+','+w['item'].lower()
        if w['description']: tmp=tmp+','+w['description'].lower()
        if w['price'] and int(w['price']) > max_pr: max_pr = int(w['price'])
    token_dict[i['name']+'_'+i['zip']] = tmp
    i['wrd_list'] = tmp
    if max_pr > 25:
        i['price_level'] = 'High'
    elif max_pr >= 10:
        i['price_level'] = 'Medium'
    else:
        i['price_level'] = 'Low'
    if int(i['avg_rating']) < 2:
        i['rating_level'] = 'Low'
    elif int(i['avg_rating']) < 3.75:
        i['rating_level'] = 'Medium'
    else:
        i['rating_level'] = 'High'

with open('dictlist.pickle', 'wb') as handle:
    pickle.dump(l, handle)

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

            ('price_lev', Pipeline([
                ('vbk', ValueByKey('price_level')),
                ('labels2', preprocessing.LabelBinarizer()),
            ])),

            ('rating_lev', Pipeline([
                ('vbk', ValueByKey('rating_level')),
                ('labels3', preprocessing.LabelBinarizer()),
            ])),

            ('nlp', Pipeline([
                ('vbk', ValueByKey('wrd_list')),
                ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english')),
            ]))
        ]
    ))
])

just_txt = Pipeline([
    ('vbk', ValueByKey('wrd_list')),
    ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english')),
])

#data=features.fit_transform(l)
data=just_txt.fit_transform(l)
tmp=features.steps[0][1].transformer_list[3][1].steps[1][1]


from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
k=12

# Initialize the kMeans cluster model.
km = KMeans(n_clusters=k, init='k-means++')

print("Clustering sparse data with %s" % km)
t0 = time()

# Pass the model our sparse matrix with the TF-IDF counts.
#km.fit(data)
labels = km.fit_predict(data)
print("done in %0.3fs" % (time() - t0))
print()

clusters = {}
n = 0
for item in labels:
    if item in clusters:
        clusters[item].append(l[n]['name'])
    else:
        clusters[item] = [l[n]['name']]
    n +=1

for item in clusters:
    print("Cluster ", item)
    for i in clusters[item]:
        print(i)


nri = {}
nri2 = {}
nri3 = {}
nri['wrd_list'] = input("Enter keywords: ")
nri2['wrd_list'] = 'sushi'
nri3['wrd_list'] = 'chicken wings, pizza'
l=[]
l.append(nri)
#l.append(nri2)
#l.append(nri3)
data=just_txt.transform(l)
labels = km.predict(data)
print(labels)






# dat = pd.DataFrame(l)
# print(dat.head())
#
# pipe = Pipeline([
#     ('encoding', preprocessing.LabelBinarizer())
# ])
#
# pipe2 = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words='english'))
# ])
#
# pipe.fit_transform(dat.type_2)
# xpipe2.fit_transform(dat.wrd_list)
#
# print(x)
# print(type(x))
# print(x2)
