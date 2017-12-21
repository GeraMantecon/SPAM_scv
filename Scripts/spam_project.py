import os
import sys
import json
import collections
import string
import re
import math
import argparse
import nltk
import gensim
import numpy as np
import pandas as pd
from parser import parse_raw_spam
from parser import parse_raw_ham
from plot import plot_clusters
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

class LabeledLineSentence(object):
    def __init__(self, documents):
        self.documents = documents
    def __iter__(self):
        for uid, line in enumerate(self.documents):
            yield gensim.models.doc2vec.LabeledSentence(words=line.split(), tags=['SENT_%s' % uid])
class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		verb_lem = [str(self.wnl.lemmatize(t,'v')) for t in word_tokenize(doc)]
		return [str(self.wnl.lemmatize(t,'n')) for t in verb_lem]
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def get_continuous_chunks(text):
	chunked = ne_chunk(pos_tag(word_tokenize(text)))
	prev = None
	continuous_chunk = []
	current_chunk = []
	for i in chunked:
		if type(i) == Tree:
			current_chunk.append(" ".join([token for token, pos in i.leaves()]))
		elif current_chunk:
			named_entity = " ".join(current_chunk)
			if named_entity not in continuous_chunk:
				continuous_chunk.append(named_entity)
				current_chunk = []
			else:
				continue
	return continuous_chunk
def generate_features(spam_messages, feature, english_only=False):
	if feature == 'POS':
		spam_features = [pos_tag(word_tokenize(spam)) for spam in spam_messages]
		if english_only:
			spam_features = [['_'.join(bigram) for bigram in spam if bigram[1] != 'FW'] for spam in spam_features]
		else:
			spam_features = [['_'.join(bigram) for bigram in spam] for spam in spam_features]
		spam_features =  [' '.join(spam) for spam in spam_features]
	elif feature == 'W2V':
		dataset = [word_tokenize(spam) for spam in spam_messages]
		model = gensim.models.Word2Vec(dataset, min_count=1, workers=4)
		model.wv.save('../Models/w2v_model.model')
		w2v = dict(zip(model.wv.index2word, model.wv.syn0))
		spam_features = w2v
	elif feature == 'D2V':
		sentences = LabeledLineSentence(spam_messages)
		model = gensim.models.Doc2Vec(sentences, window=8, min_count=5, workers=4)
		model.wv.save('../Models/d2v_model.model')
		spam_features = model
	elif feature == 'NAMED_ENTITIES':
		spam_features = [get_continuous_chunks(spam) for spam in spam_messages]
		spam_features = [spam for spam in spam_messages if spam]
	else:
		spam_features = spam_messages
	return spam_features
def print_labeled_tfidfs(tfidfs,messages,vectorizer,labels,k,feature,spam=False):
    tfidf = pd.DataFrame(tfidfs.toarray(),index=[index for index, value in enumerate(messages)],columns=vectorizer.get_feature_names())
    if k >= 0:
        new_df = pd.DataFrame(data=labels, index=[index for index, value in enumerate(labels)], columns=['Label'])
    else:
        new_df = pd.DataFrame(data=[-1 for label in labels], index=[index for index, value in enumerate(labels)], columns=['Label'])
    final_df = pd.merge(tfidf,new_df,left_index=True,right_index=True)
    if spam:
        final_df.to_csv('../Results/Spam'+feature+'_'+str(k)+'.csv',index=False)
    else:
        final_df.to_csv('../Results/Ham'+feature+'_'+str(k)+'.csv',index=False)
def train_models_clustering(spam_messages,spam_features,feature,k):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, verbose=0)
    if feature == 'BOW':
        vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,1),lowercase=True)
        tfidfs = vectorizer.fit_transform(spam_messages)
        results = km.fit_transform(tfidfs)
    elif feature == 'BIGRAMS':
        vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(2,2),lowercase=True)
        tfidfs = vectorizer.fit_transform(spam_messages)
        results = km.fit_transform(tfidfs)
        #print_labeled_tfidfs(tfidfs,spam_messages,vectorizer,km.labels_,k,feature,True)
    elif feature == 'TRIGRAMS':
        vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(3,3),lowercase=True)
        tfidfs = vectorizer.fit_transform(spam_messages)
        results = km.fit_transform(tfidfs)
        #print_labeled_tfidfs(tfidfs,spam_messages,vectorizer,km.labels_,k,feature,True)
    elif feature == 'W2V':
        vectorizer = TfidfEmbeddingVectorizer(spam_features)
        tfidfs = vectorizer.fit(spam_messages).transform(spam_messages)
        results = km.fit_transform(tfidfs)
        #print_labeled_tfidfs(tfidfs,spam_messages,vectorizer,km.labels_,k,feature,True)
    elif feature == 'D2V':
        spam_vectors = spam_features.wv.syn0
        results = km.fit_transform(spam_vectors)
        #print_labeled_tfidfs(tfidfs,spam_messages,vectorizer,km.labels_,k,feature,True)
    elif feature == 'NAMED_ENTITIES':
        vocabulary = [token for spam in spam_features for token in word_tokenize(spam)]
        vocabulary = list(set(vocabulary))
        vectorizer = TfidfVectorizer(use_idf=True, vocabulary = vocabulary, tokenizer=lambda i:i, lowercase=False)
        tfidfs = vectorizer.fit_transform(spam_messages)
        results = km.fit_transform(tfidfs)
    elif feature == 'POS':
        vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,1), lowercase=True)
        tfidfs = vectorizer.fit_transform(spam_messages)
        results = km.fit_transform(tfidfs)
    else:
        print 'ERROR: FEATURE NOT SUPPORTED.'
        results = []
        labels = []
    return results,km.labels_
def pre_process(stopword=False,punctuation=False,lematize=False,isSpam=False):
    if isSpam:
        path='../Resources/spam.txt'
    else:
        path='../Resources/ham.txt'
    print path
    with open(path) as f:
		spams = f.readlines()
    spams = [eval(spam)['body'] for spam in spams]
    if punctuation:
        spams = [" ".join([re.sub(r'[^\w]', '', token) for token in spam.lower().strip().split() if token not in set(string.punctuation)]) for spam in spams]
    if stopwords:
        spams = [" ".join([token for token in spam.lower().strip().split() if token not in set(stopwords.words("english"))]) for spam in spams]
    if lematize:
        tk = LemmaTokenizer()
        spams = [" ".join(tk(spam.lower().strip())) for spam in spams]
    return spams
def main(argv):
    parser = argparse.ArgumentParser(description='SCV Spam Classification Viaduct')
    parser.add_argument('FEATURE', type=str, help='Desired feature upon which the clustering algorithm is goin to be trained: POS, BOW, BIGRAMS, TRIGRAMS, W2V, NAMED_ENTITIES')
    parser.add_argument('k', type=int ,help='K number for K-mean')
    parser.add_argument('-eo','--english', action ='store_true', help='Filter to work only with english.')
    parser.add_argument('-sw','--stopword', action ='store_true', help='Remove stopwords.')
    parser.add_argument('-p','--punctuation', action ='store_true', help='Remove punctuation marks.')
    parser.add_argument('-l', '--lematize', action ='store_true',help='Lemmatize tokens.')
    parser.add_argument('-m', '--models', action ='store_true',help='Use already trained model on Models directory.')
    args = parser.parse_args()
    if args.FEATURE not in ['POS','BOW','BIGRAMS','TRIGRAMS','W2V','NAMED_ENTITIES','D2V']:
        print 'Feature not recognized by program.'
        sys.exit()
    if not os.path.isfile('../Resources/spam.txt'):
        parse_raw_spam()
    if not os.path.isfile('../Resources/ham.txt'):
        parse_raw_ham()
    spam_messages = pre_process(args.stopword, args.punctuation, args.lematize,isSpam=True)
    spam_features = generate_features(spam_messages,args.FEATURE,args.english)
    results,labels = train_models_clustering(spam_messages,spam_features,args.FEATURE,args.k)
    score = metrics.silhouette_score(results, labels, metric='euclidean')
    plot_clusters(results,args.k,labels,args.FEATURE)
    print 'K-means with '+ str(args.k) +' clusters using '+ str(args.FEATURE) +' silhouette score: ' + str(score)
if __name__ == '__main__':
    main(sys.argv[1:])
