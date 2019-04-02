# Orthopedic Patients Prediction Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Orthopedic_patients.csv')
X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.fit_transform(y_test)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

predictions = []
for i in range(len(y_pred)):
    if(y_pred[i] > 0.7):
        predictions.append(1)
    else:
        predictions.append(0)

predictions = np.array(predictions,dtype=np.int64)
        
from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test, predictions) * 100,"%")