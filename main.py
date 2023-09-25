#import necessary modules
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading dataset
dataset = pd.read_csv("SONAR_rock_vs_mine_prediction/sonar.all-data.csv", header=None)
print(dataset.head())
print(dataset.tail())

#no of rows and columns
print(dataset.shape)

#describe statistical measures of the data
print(dataset.describe())

print(dataset[60].value_counts())

print(dataset.groupby(60).mean())

#separating data and labels
X = dataset.drop(columns=60, axis = 1)
Y = dataset[60]
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2) #stratify --> we need almost equal no.of rock and mine in training data.
print(X.shape, X_train.shape, X_test.shape)

#model training
model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("accuracy of training data", training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("accuracy of training data", testing_data_accuracy)

#making predictive system
input_data = (0.0065,0.0122,0.0068,0.0108,0.0217,0.0284,0.0527,0.0575,0.1054,0.1109,0.0937,0.0827,0.0920,0.0911,0.1487,0.1666,0.1268,0.1374,0.1095,0.1286,0.2146,0.2889,0.4238,0.6168,0.8167,0.9622,0.8280,0.5816,0.4667,0.3539,0.2727,0.1410,0.1863,0.2176,0.2360,0.1725,0.0589,0.0621,0.1847,0.2452,0.2984,0.3041,0.2275,0.1480,0.1102,0.1178,0.0608,0.0333,0.0276,0.0100,0.0023,0.0069,0.0025,0.0027,0.0052,0.0036,0.0026,0.0036,0.0006,0.0035)

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data) #Unlike Python lists, NumPy arrays can only contain elements of the same data type (e.g., integers, floats, etc.). This homogeneity allows for more efficient numerical operations.

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) #The first argument 1 indicates that you want to have one row in the reshaped array. This is because you are making a prediction for a single instance or data point, and you want to represent it as one row.
                                                               #The second argument -1 is a placeholder that tells NumPy to automatically calculate the number of columns needed to maintain the original data's shape. In other words, NumPy will figure out how many columns are required based on the size of your original data.

'''Let's say your original input_data had 60 elements. When you reshape it using reshape(1, -1),
NumPy will create a 2D array with one row and 60 columns, preserving the original data's structure.
If you had 100 elements, it would create a 2D array with one row and 100 columns.'''

prediction = model.predict(input_data_reshaped)
print(prediction)
'''converting your input data into a NumPy array is a best practice when working with numerical data in Python, especially when dealing with machine learning tasks. It enhances efficiency, compatibility with machine learning libraries, and provides the necessary tools for data manipulation and numerical operations.'''

if(prediction[0] == 'R'):
    print("The object is a Rock")
else:
    print("The object is a Mine")