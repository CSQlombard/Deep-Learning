import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import urllib2

## Run deep Learning in keras
#predictors = np.loadtxt(‘predictors_data_csv’, delimeter=’,’)

predictors = total_concept_matrix
n_cols = predictors.shape[1]

# Convert the target to categorical: target

url = 'https://s3.eu-central-1.amazonaws.com/deeplearningbookssebidata/Deep_Learning_Books_Data/ytrain.txt'
file = urllib2.urlopen(url)
ytest = np.loadtxt(file)
ytest = ytest[0:total_concept_matrix.shape[0]]

target_final = to_categorical(ytest)

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(target_final.shape[1], activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
hist = model.fit(predictors, target_final,epochs=30, validation_split = 0.3)

# Calculate predictions: for this I have to convert also the test.
predictions = model.predict(predictors)
predicted_prob_true = predictions[:,1]
