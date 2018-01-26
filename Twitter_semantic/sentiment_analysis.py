
# coding: utf-8

# In[1]:

#VOTING

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import pickle

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes)) #count how many occurences of most popular votes.
        conf = choice_votes / len(votes)
        return conf


documents_f = open("C:\\Data_jupyter\\pickled_algos\\documents.pickle","rb")
document=pickle.load(documents_f)
documents_f.close()


# In[2]:

word_feature_f = open("C:\\Data_jupyter\\pickled_algos\\word_features5k.pickle","rb")
word_features = pickle.load(word_feature_f)
word_feature_f.close()

def find_features(document):
    words=word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# In[3]:

open_features = open("C:\\Data_jupyter\\pickled_algos\\feature_set.pickle","rb")
featuresets=pickle.load(open_features)
open_features.close()


random.shuffle(featuresets)

#only positive testing set
training_set = featuresets[:10000]
testing_set = featuresets[10000:]
print(len(featuresets))



# In[5]:

classifier_open=open("C:\\Data_jupyter\\pickled_algos\\originalnaivebayes5k.pickle","rb")
classifier = pickle.load(classifier_open)
classifier_open.close()

open_file = open("C:\\Data_jupyter\\pickled_algos\\MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("C:\\Data_jupyter\\pickled_algos\\BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("C:\\Data_jupyter\\pickled_algos\\Logistic_Regression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("C:\\Data_jupyter\\pickled_algos\\LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("C:\\Data_jupyter\\pickled_algos\\SGDClassifier_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


# In[6]:

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


# In[8]:

#print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))

#print(sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))

