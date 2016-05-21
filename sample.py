from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

with open('dictlist.pickle', 'rb') as handle:
    l = pickle.load(handle)

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

data=features.fit_transform(l)
tmp=features.steps[0][1].transformer_list[3][1].steps[1][1]

from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
k=10

# Initialize the kMeans cluster model.
km = KMeans(n_clusters=k, init='k-means++', max_iter=100)

print("Clustering sparse data with %s" % km)
t0 = time()

# Pass the model our sparse matrix with the TF-IDF counts.
km.fit(data)
print("done in %0.3fs" % (time() - t0))
print()

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = features.steps[0][1].transformer_list[3][1].steps[1][1].get_feature_names()
for i in range(k):
    print("Cluster %d:" % (i+1), end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()



################################################################################
## Below is my manual approach to getting unique sets of keywords per restaurant and overall
## Load the final list of all restaurant objects before conducting NLP on each one
# with open('loaderfinalrlist.pickle', 'rb') as handle: l = pickle.load(handle)
#
# stop_words = set(stopwords.words('english'))
# master_word_list = set()
#
# for i in l:
#     i['word_chunks'] = set()
#     print(i['name'])
#     itm = process_chunks(i, 'item')
#     dsc = process_chunks(i, 'description')
#
#     for v in itm:
#         master_word_list.add(v)
#         i['word_chunks'].add(v)
#     for v in dsc:
#         master_word_list.add(v)
#         i['word_chunks'].add(v)
#
# print(master_word_list)
# print(l[1]['name'])
# print(l[1]['word_chunks'])
# print(l[5]['name'])
# print(l[5]['word_chunks'])
# print("Number of keywords: ",len(master_word_list))
################################################################################
