import util
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data_dir = util.data_dir
input_dir = util.input_dir

## loading the files

# df_rage = pd.read_csv(os.path.join(data_dir,'processed_rage.csv'))
# df_happy =  pd.read_csv(os.path.join(data_dir,'processed_happy.csv'))
# df_gore =  pd.read_csv(os.path.join(data_dir,'processed_gore.csv'))
# df_creepy =  pd.read_csv(os.path.join(data_dir,'processed_creepy.csv'))


## loading the filtered data

class Preprocessing:

	def __init__(self):
		self.data = self.get_filtered_data()
		self.le = preprocessing.LabelEncoder()
		train_data, test_data, Y_train, Y_test = self.get_train_test()
		self.train_data = train_data
		self.test_data = test_data
		self.Y_train = Y_train
		self.Y_test = Y_test


	def get_filtered_data(self):
		df_input = pd.read_csv(os.path.join(input_dir,"input_data.csv"))
		length = np.min(list(df_input.subreddit.value_counts()))
		## creating a balanced dataset
		df_happy =(df_input.loc[df_input.subreddit == "happy"])
		df_gore = (df_input.loc[df_input.subreddit == "gore"])
		df_rage = (df_input.loc[df_input.subreddit == "rage"])
		df_creepy = (df_input.loc[df_input.subreddit == "creepy"])
		df_input = pd.concat([ df_happy[:length], df_rage[:length], df_gore[:length], df_creepy[:length]], ignore_index=True)
		return df_input[:10]

	def get_train_test(self):
		le = self.le
		df_input = self.data
		Y = df_input['subreddit']
		le.fit(Y)
		print("output categories :", le.classes_)
		y = le.transform(Y)
		print("splitting the dataset into train and test")
		train_data, test_data, Y_train, Y_test = train_test_split(df_input, y, test_size=0.20, random_state=6)
		return (train_data, test_data, Y_train, Y_test)