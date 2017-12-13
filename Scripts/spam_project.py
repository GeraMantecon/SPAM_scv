import os
import sys
import json
import collections
import string
import re
import math
import argparse
import nltk
import enchant
import gensim
from parser import parse_raw_spam
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		verb_lem = [str(self.wnl.lemmatize(t,'v')) for t in word_tokenize(doc)]
		return [str(self.wnl.lemmatize(t,'n')) for t in verb_lem]

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
def generate_features(spam_messages, feature, english_only=False,):
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
		spam_features = model
	elif feature == 'NAMED_ENTITIES':
		spam_features = [get_continuous_chunks(spam) for spam in spam_messages]
		spam_features = [spam for spam in spam_messages if spam]
		for i in range(5):
			print spam_features[i]
	else:
		print 'ERROR: FEATURE NOT SUPPORTED.'
	return spam_features
def train_models(spam_features):
	pass
def pre_process(stopword=False,punctuation=False,lematize=False):
	with open('../Resources/spam.txt') as f:
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
	parser.add_argument('-eo','--english', action ='store_true', help='Filter to work only with english.')
	parser.add_argument('-sw','--stopword', action ='store_true', help='Remove stopwords.')
	parser.add_argument('-p','--punctuation', action ='store_true', help='Remove punctuation marks.')
	parser.add_argument('-l', '--lematize', action ='store_true',help='Lemmatize tokens.')
	parser.add_argument('-m', '--models', action ='store_true',help='Use already trained model on Models directory.')
	args = parser.parse_args()
	if args.FEATURE not in ['POS','BOW','BIGRAMS','TRIGRAMS','W2V','NAMED_ENTITIES']:
		print 'Feature not recognized by program.'
		sys.exit()
	if not os.path.isfile('../Resources/spam.txt'):
		parse_raw_spam()
	spam_messages = pre_process(args.stopword, args.punctuation, args.lematize)
	if not args.models:
		spam_features = generate_features(spam_messages,args.FEATURE,args.english)
		#traine_models(spam_features)

if __name__ == '__main__':
    main(sys.argv[1:])
