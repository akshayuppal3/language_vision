
## join the filtered tokens back again to the keras tokenizer which would give vocalb words etc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import util
import os
import numpy as np
import pickle
from keras.models import Model, Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed

embedding_path = os.path.join(util.embedding_dir,"fasttext_100.vec")
model_dir = util.model_dir

class Language:
	def __init__(self,data):
		token_list = (data['title'].apply(util.get_tokens))
		self.max_length = self.get_max_length(token_list)
		sentences= [' '.join(tokens) for tokens in token_list]
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(sentences)
		self.tokenizer = tokenizer
		self.vocab =  len(tokenizer.word_index) + 1
		self.word2vec = self.get_word2vec(embedding_path)
		self.embedding_matrix = self.get_embedding_matrix()
		self.lng_model = self.get_bilstm_model()

	def get_max_length(self,token_list):
		max_len = 0
		for idx, tokens in enumerate(token_list):
			if len(tokens) > max_len:
				max_len = len(tokens)
		return max_len

	def get_encoded_data(self,tokens):
		senetences = [' '.join(tokens) for tokens in tokens]
		# integer encode the documents
		encoded_docs = self.tokenizer.texts_to_sequences(senetences)
		# pad documents to a max length of 4 words
		lng_data = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
		return lng_data

	def get_word2vec(self,file_path):
		file_w2v = os.path.join(model_dir,"w2v.pkl")
		exists = os.path.isfile(file_w2v)
		if exists:                                      # load a prexisting pickel module
			word2vec = pickle.load(open(file_w2v,"rb"))
			return word2vec
		else:
			file = open(file_path, "r")
			if (file):
				word2vec = dict()
				split = file.read().splitlines()
				for line in split:
					key = line.split(' ', 1)[0]  # the first word is the key
					value = np.array([float(val) for val in line.split(' ')[1:]])
					word2vec[key] = value
				util.dump(word2vec,os.path.join(util.model_dir,"w2v.pkl"))
				print("dumped the w2v file")
				return (word2vec)

	def get_embedding_matrix(self):
		from numpy import zeros
		embedding_matrix = zeros((self.vocab,100))
		for word, i in self.tokenizer.word_index.items():
			embedding_vector = self.word2vec.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
			return embedding_matrix


	def get_bilstm_model(self):
		max_len = self.max_length
		embedding_matrix = self.embedding_matrix
		input1 = Input(shape=(max_len,))
		model = Embedding(self.vocab, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)(
			input1)
		model = Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.25), merge_mode="concat")(model)
		model = TimeDistributed(Dense(100, activation="relu"))(model)
		model = Flatten()(model)
		model = Dense(100, activation='relu')(model)
		out = Dense(4, activation='softmax')(model)
		lng_model = Model(inputs=input1, outputs=out)
		lng_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(lng_model.summary())
		return lng_model



