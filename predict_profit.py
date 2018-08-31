# Importing the Pandas Library
import pandas as pd

# Using pandas library to load the data.
dataset = pd.read_csv('Startups_list.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])
oneHotEnc = OneHotEncoder(categorical_features = [3])
X = oneHotEnc.fit_transform(X).toarray()

# Split the dataset into training set and test set so that we can train our model with our training set
from sklearn.cross_validation import train_test_split
training_input, testing_input, training_output, testing_output= train_test_split(X, y, test_size = 0.2, random_state = 0)

# Using Multiple Linear Regression on our dataset by importing the LinearRegression class and creating an object from it
# and fitting our training set into the model.
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(training_input,training_output)

# Using the trained model to predict the output of the test set by using the predict method in LinearRegression
predictedOutput = regression.predict(testing_input)

# The predicted output is compared with the test set output and accuracy can be checked.
