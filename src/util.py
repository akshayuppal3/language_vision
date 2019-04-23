import pandas as pd
import json
import os
from pathlib import Path
import posixpath
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet as wn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


stopwords = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('stopwords')

## loading the config file
dir_name = os.getcwd()
path1 = str(Path(os.getcwd()).parent)
filepath = posixpath.join(path1, 'config.json')
tweet_tokenizer = TweetTokenizer()

with open(filepath) as f:
	data = json.load(f)

## loading the different configuration files
data_dir = os.path.join(path1, data['data_dir'])
embedding_dir = os.path.join(path1, data['embedding_dir'])
model_dir = os.path.join(path1, data['model_dir'])
input_dir = os.path.join(path1, data['input_dir'])


def get_lemma(word):
	lemma = wn.morphy(word)
	if lemma is None:
		return word
	else:
		return lemma


def get_tokens(sentence):
	tknzr = TweetTokenizer()
	tokens = tknzr.tokenize(sentence)
	tokens = [token for token in tokens if (token not in stopwords and len(token) > 1)]
	tokens = [get_lemma(token) for token in tokens]
	return (tokens)


def dump(data, filepath):
	with open(filepath, "wb") as f:
		pickle.dump(data, f)


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def PCA_with_SVM(X_train, Y_train):
	estimators = [('reduce_dim', PCA()), ('clf', SVC())]
	pipe = Pipeline(estimators)
	pipe.fit(X_train, Y_train)
	return (pipe)


def svm_wrapper(X_train, Y_train):
	param_grid = [
		{'C': [1, 10], 'kernel': ['linear']},
		{'C': [1, 10], 'gamma': [0.1, 0.01], 'kernel': ['rbf']}, ]
	svm = GridSearchCV(SVC(), param_grid)
	svm.fit(X_train, Y_train)
	return (svm)


def radom_forest_wrapper(X_train, Y_train):
	param_grid = [
		{'max_depth': [i for i in np.arange(1, 10)],
		 'n_estimators': [j for j in np.arange(1, 10)]}]
	rand = GridSearchCV(RandomForestClassifier(), param_grid)
	rand.fit(X_train, Y_train)
	return (rand)

def naive_bayes(X_train, Y_train):
	clf = BernoulliNB()
	clf.fit(X_train, Y_train)
	return (clf)


def mlp_wrapper(X_train, Y_train):
	param_grid = [
		{'hidden_layer_sizes': np.arange(10, 100, 4), 'max_iter': [30], 'activation': ['logistic', 'tanh', 'relu']}]
	mlp = GridSearchCV(MLPClassifier(), param_grid)
	mlp.fit(X_train, Y_train)
	return (mlp)


def boosting(X_train, Y_train):
	gb = GradientBoostingClassifier()
	gb.fit(X_train, Y_train)
	return (gb)
