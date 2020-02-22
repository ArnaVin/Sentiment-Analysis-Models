import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf
'''
short_pos = open("dataSet/positive.txt","r").read()
short_neg = open("dataSet/negative.txt","r").read()

documents = []
all_words = []

#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p,"pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p,"neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled_stuff/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open("pickled_stuff/wordF5K.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()
'''
documents_f = open("pickled_stuff/documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

wordF5K = open("pickled_stuff/wordF5K.pickle","rb")
word_features = pickle.load(wordF5K)
wordF5K.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in  documents]
random.shuffle(featuresets)
#print("Featuresets :", len(featuresets))

training_set = featuresets[:10000]
testing_sets = featuresets[10000:]

#NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
NB_classifier_f = open("pickled_stuff/naiveBayes.pickle","rb")
NB_classifier = pickle.load(NB_classifier_f)
NB_classifier_f.close()

#print("NB_classifier accuracy    : ", (nltk.classify.accuracy(NB_classifier, testing_sets))*100)
'''
save1_classifier = open("pickled_stuff/naiveBayes.pickle","wb")
pickle.dump(NB_classifier, save1_classifier)
save1_classifier.close()
'''
#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(training_set)
MNB_classifier_f = open("pickled_stuff/MultinomialNB.pickle","rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()

#print("MNB_classifier accuracy   : ", (nltk.classify.accuracy(MNB_classifier, testing_sets))*100)
'''
save2_classifier = open("pickled_stuff/MultinomialNB.pickle","wb")
pickle.dump(MNB_classifier, save2_classifier)
save2_classifier.close()
'''
#BNB_classifier = SklearnClassifier(BernoulliNB())
#BNB_classifier.train(training_set)
BNB_classifier_f = open("pickled_stuff/BernoulliNB.pickle","rb")
BNB_classifier = pickle.load(BNB_classifier_f)
BNB_classifier_f.close()

#print("BNB_classifier accuracy   : ", (nltk.classify.accuracy(BNB_classifier, testing_sets))*100)
'''
save3_classifier = open("pickled_stuff/BernoulliNB.pickle","wb")
pickle.dump(BNB_classifier, save3_classifier)
save3_classifier.close()
'''
#SGD_classifier = SklearnClassifier(SGDClassifier())
#SGD_classifier.train(training_set)
SGD_classifier_f = open("pickled_stuff/SGDClassifier.pickle","rb")
SGD_classifier = pickle.load(SGD_classifier_f)
SGD_classifier_f.close()

#print("SGD_classifier accuracy   : ", (nltk.classify.accuracy(SGD_classifier, testing_sets))*100)
'''
save4_classifier = open("pickled_stuff/SGDClassifier.pickle","wb")
pickle.dump(SGD_classifier, save4_classifier)
save4_classifier.close()
'''

voted_classifier = VoteClassifier(NB_classifier, MNB_classifier, BNB_classifier)
#print("VOTED_classifier accuracy : ", (nltk.classify.accuracy(voted_classifier, testing_sets))*100)
'''
print("Classification:", voted_classifier.classify(testing_sets[0][0]), "Confidence:",voted_classifier.confidence(testing_sets[0][0])*100)
print("Classification:", voted_classifier.classify(testing_sets[1][0]), "Confidence:",voted_classifier.confidence(testing_sets[1][0])*100)
print("Classification:", voted_classifier.classify(testing_sets[2][0]), "Confidence:",voted_classifier.confidence(testing_sets[2][0])*100)
print("Classification:", voted_classifier.classify(testing_sets[3][0]), "Confidence:",voted_classifier.confidence(testing_sets[3][0])*100)
print("Classification:", voted_classifier.classify(testing_sets[4][0]), "Confidence:",voted_classifier.confidence(testing_sets[4][0])*100)
'''
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats),NB_classifier.classify(feats),BNB_classifier.classify(feats),MNB_classifier.classify(feats),SGD_classifier.classify(feats)
