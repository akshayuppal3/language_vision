# Launguage_vision
Project related to multimodal classification of emotion utilizing langugage and image modalities. It was part of CS-594  course (under Prof. Natalie Parde) requirement at UIC

Details included in the report

Installation Intructions:
1)	Extract the zip file in root/
2)	Navigate to root/language_vision directory
3)	Create a virtual env with conda  (assuming the anaconda is already installed)
a.	conda env create -f environment.yml
4)	Activate the environment 
a.	Source activate multimodal-env
5)	Install the glove embeddings (glove.twitter.27B.zip) from                                                                        (http://nlp.stanford.edu/data/glove.twitter.27B.zip) 
6)	Extract the zip and place the glove.twitter.27B.100d  (100-Dimension) file in the root/language_vision/embeddings  directory
7)	Reddit dataset is already filtered and present in the input directory (input_data.csv)
8)	Navigate to the  root/language_vision/src directory
9)	To run the code 
a.	python3  driver.py                    
10)	Unimodal and fusion results will be printed in the console

Some of common issues:
1)	You might encounter memory error for creating word2vec for language model as conversion of binary file to dictionary tend to take up lot of space [2-3GB approx.]

Anaconda installation
•	https://docs.anaconda.com/anaconda/install/

 

References:

[1]Duong, Chi Thang, Remi Lebret, and Karl Aberer. "Multimodal classification for analysing social media." arXiv preprint arXiv:1708.02099 (2017).
