from preprocessing import Preprocessing
from language import Language
from image import Image_modality
import util
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.models import Model, Input
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
	pre = Preprocessing()
	df_input = pre.data
	train_data = pre.train_data
	test_data = pre.test_data
	Y_train = pre.Y_train
	Y_test = pre.Y_test


	lng = Language(df_input)

	print("Preparing the language data")
	train_tokens = train_data['title'].apply(util.get_tokens)
	lng_data_train = lng.get_encoded_data(train_tokens)

	test_tokens = test_data['title'].apply(util.get_tokens)
	lng_data_test = lng.get_encoded_data(test_tokens)
	language_model = lng.lng_model
	print("training the language model (bi-lstm), this might take some time")
	language_model.fit(lng_data_train, Y_train, verbose=1, validation_split=0.2, nb_epoch=5)

	## printing precision_recall- language modality
	Y_pred = language_model.predict(lng_data_test, verbose=1)
	y_pred = np.array([np.argmax(pred) for pred in Y_pred])
	print("******************language modality scores(unimodal)*******************************")
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')


	image_mod = Image_modality()

	target_size = (224,224)  ## target size for resnet
	print("preparing the image data")
	image_data_train = image_mod.get_image_input(train_data,target_size)
	image_data_test = image_mod.get_image_input(test_data,target_size)

	image_model = image_mod.get_resnet50()
	print("training the image model.. might take some time")
	image_model.fit(image_data_train, Y_train, verbose=1, nb_epoch=15, validation_split=0.2)

	## printing precision_recall - image modality
	Y_pred = image_model.predict(image_data_test, verbose=1)
	y_pred = np.array([np.argmax(pred) for pred in Y_pred])
	print("******************image modality scores(unimodal)*******************************")
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')

	print ("finally fusing the vectors from each languge and image model")

	model_1 = Model(input=image_model.input, output=image_model.get_layer('dense_30').output)       ## based on the name of resnet50 layer
	model_2 = Model(input=language_model.input, output=language_model.get_layer('dense_38').output) ## based on the name of bilstm model

	print(" getting the individual vecotrs for image and language")
	print(" getting the fused training vecotrs")
	part1_train = model_1.predict(image_data_train, verbose=1)
	part2_train = model_2.predict(lng_data_train, verbose=1)

	X_train =  np.hstack((part1_train,part2_train))

	print(" getting the fused testing vectors")
	part1 = model_1.predict(image_data_test, verbose=1)
	part2 = model_2.predict(lng_data_test, verbose=1)

	X_test = np.hstack((part1, part2))

	print("done fusing the vectors")
	## getting the scores with late fusion
	print ("getting the metrics (Precision/Recall) with late fusion")
	print("******************fusing modality scores(multimodal)*******************************")
	# svm
	svm = util.svm_wrapper(X_train, Y_train)
	Y_pred = svm.predict(X_test)
	score = accuracy_score(Y_test, Y_pred)

	print("accuarcy :", score)
	confusion_matrix(Y_test, Y_pred)
	print('  Classification Report:\n', classification_report(Y_test, Y_pred), '\n')

