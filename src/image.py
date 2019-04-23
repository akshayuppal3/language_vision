import urllib.request
import os
import pandas as pd
from io import BytesIO
import numpy as np
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import pickle
from keras.layers import Lambda, Input
from keras.models import Model
from keras.backend import tf as ktf
import urllib.request
from requests.exceptions import ConnectionError
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_incept
from keras.layers import MaxPooling1D
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_res
from keras.layers import Maximum
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from keras.models import Model
import numpy as np
import urllib3
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

## trying with img1
# @ param: img url @ return feature vector
from PIL import Image
import requests


class Image_modality:

	def get_img_data(self,url, size):
		try:
			response = requests.get(url)
			img_data = 0
			if (response.status_code == 200):
				try:
					if (url.endswith('.jpg') or url.endswith('.jpeg') or url.endswith('.png')):
						with urllib.request.urlopen(url) as url1:
							with open('temp.jpg', 'wb') as f:
								f.write(url1.read())
						fh = open('temp.jpg')
						if fh:
							img = image.load_img('temp.jpg', target_size=size)  # 224*224
							os.remove('temp.jpg')
							img_data = image.img_to_array(img)
						return img_data
				except RuntimeError as e:
					return 0  # except runtime error fro file that doesn't exits
			return img_data

		except FileNotFoundError as e:
			return 0
		except IOError:
			return 0
		except ConnectionError as e:
			return 0
		except urllib.error.HTTPError as e:
			return 0

	def get_resnet50(self):
		model = ResNet50(weights='imagenet', include_top=True)
		model_res = Model(input=model.input, output=model.get_layer('fc1000').output)
		input = model_res.output
		model = Dense(100,activation= 'relu',name='dense_30')(input)  ## specifying name for fusion model
		out = Dense(4, activation='softmax')(model)
		model_img = Model(input=model_res.input, output=out)
		for layer in model_res.layers:                          ## freeze the keras layers
			layer.trainable = False
		model_img.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model_img

	## load the image data
	def get_image_input(self,df, size):
		img_data = list()
		for index, row in tqdm(df.iterrows(), total=len(df)):
			temp = self.get_img_data(row['url'], size)
			img_data.append(temp)
		img_data = np.array(img_data)
		img_data = preprocess_input_res(np.array(list(img_data)))
		return img_data