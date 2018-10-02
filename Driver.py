from Iris import Iris
import numpy as np

print ("Iris Class Predictor")
print ("Attributes\nsepal length\nsepal width\npetal length\npetal width\n\n\nClass: Iris Setosa\tIris Versicolour\t Iris Virginica")

user_input = []
print ("User Input \nExamples\n5.1,3.5,1.4,0.2\n4.9,3.0,1.4,0.2")

print ("Sepal length")
user_input.append(float(input()))
print ("Sepal Width")
user_input.append(float(input()))
print ("Petal length")
user_input.append(float(input()))
print ("Petal Width")
user_input.append(float(input()))

input_2d_format = [user_input]

iris = Iris()
predictions = iris.KNN(input_2d_format)
iris.displayPrediction(input_2d_format, predictions)