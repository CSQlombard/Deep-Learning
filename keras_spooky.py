import numpy as np
import pandas as pd
import io
#import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D

## Define training data
"""
predictors = concept_matrix[0:dim_train,]
n_cols = predictors.shape[1]
predictors = doc_term_mat[0:dim_train,]


predictors = matrix_transf[0:dim_train,]
predictors = predictors.toarray()
#n_cols = predictors.shape[1]

predictors = matrix_transf[0:dim_train,]
n_cols = matrix_transf.shape[1]

"""
# Get the test data for the training
def transform_test(dim_train):
    ytest = np.empty([dim_train,1],dtype=unicode)
    file = io.open('train.csv','r',encoding='utf-8')
    c = 0
    for index, info in enumerate(file.readlines()):
        if index > 0: ## first line are labels
            info = info.split('","')
            if info[2] == u'EAP"\n':
                ytest[index-1] = int(0)
                c = c + 1
            if info[2] == u'HPL"\n':
                ytest[index-1] = int(1)
                c = c + 1
            if info[2] == u'MWS"\n':
                ytest[index-1] = int(2)
                c = c + 1

    if c != dim_train:
        print("Mistake!!")

    target_final = to_categorical(ytest)
    return target_final

"""
## Run Deep Learning Algorithm
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
#model.add(Dropout(0.5))
#model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(target_final.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
hist = model.fit(predictors, target_final,nb_epoch=3, validation_split = 0.3)

sequence_input = Input(shape=(predictors.shape[1],), dtype='int32')
#embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(sequence_input)
x = MaxPooling1D(5)(x)
reds = Dense(target_final.shape[1], activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(predictors, target_final,nb_epoch=30, validation_split = 0.3)

test_pred = matrix_transf[dim_train:,]
predictions = model.predict(test_pred.toarray())
"""

## Compute prediction and save submission
# 1500, stem, 2 neurons, 0.4892
# 2000, stem, 2 neurons, 0.4777

def ident_func(predictions):
    file = io.open('test.csv','r',encoding='utf-8')
    ident = []
    for index, info in enumerate(file.readlines()):
        info = info.split('","')
        if index > 0: ## first line are labels
            ident.append(str(info[0][1:]))

    ident = pd.DataFrame(ident, columns = ['id'])
    df = pd.DataFrame(predictions,columns = ['EAP','HPL','MWS'])
    dff = [ident, df]
    result = pd.concat(dff,axis=1)
    result.to_csv('my_submission_14_12_2017.csv', sep=',',index=False)

# Save data
#ident = pd.DataFrame(ident, columns = ['id'])
#df = pd.DataFrame(predictions,columns = ['EAP','HPL','MWS'])
#dff = [ident, df]
#result = pd.concat(dff,axis=1)
#result.to_csv('my_submission_20_11_2017.csv', sep=',',index=False)
#scp -i boot.pem ubuntu@ec2-52-16-20-62.eu-west-1.compute.amazonaws.com:~/data/my_submission_19_11_2017.csv ~/Dokumente/Deep_Learning/Spooky_Author
