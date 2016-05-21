import nltk
import re
import pprint
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


#getty = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this."
getty = "heirloom tomatoes, meat plate, hot dogs, spaghetti"

stop_words = set(stopwords.words('english'))

tokens = nltk.word_tokenize(getty.lower())

tagged=nltk.pos_tag(tokens)
chunkGram = r"""Chunk: {<N.+>*}"""
chunkParser = nltk.RegexpParser(chunkGram)
chunked = chunkParser.parse(tagged)
wrds=[]
for subtree in chunked.subtrees():
    #print(subtree.leaves())
    wrds.append(" ".join([i[0] for i in subtree.leaves()]))

print(wrds[1:])

'''
try:
    tagged = nltk.pos_tag(tokens)
    #chunkGram = r"""Chunk: {<NN.?>*<NN.*>?}"""
    chunkGram = r"""Chunk: {<N.+>*}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    #print(chunked)
    #chunked.draw()
except Exception as e:
    print(str(e))

for subtree in chunked.subtrees():
    #print(subtree.leaves())
    wrds.append(" ".join([i[0] for i in subtree.leaves()]))

print(wrds[1:])


#filtered = [w for w in tokens if not w in stop_words]
#tagged = nltk.pos_tag(filtered)

#print(tagged)
'''
