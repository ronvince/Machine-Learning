import pandas
import numpy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Iris:
	def __init__(self):
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
		names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
		dataset = pandas.read_csv(url, names=names)	
		array = dataset.values
		self.X_train = array[:,0:4]
		self.Y_train = array[:,4]
		

	def KNN(self, input_2d_format =[[]]):
		knn = KNeighborsClassifier()
		knn.fit(self.X_train, self.Y_train)
		predictions = knn.predict(input_2d_format)
		return predictions

	def displayPrediction(self, input_2d_format=[[]], predictions = []):
		print (input_2d_format)
		print (predictions)
		