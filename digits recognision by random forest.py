# -*- coding: utf-8 -*-
#%% Libraries
from os import listdir
import numpy as np
from PIL import Image
import pandas as pd


# sorting function
def sortByLength(inputStr):
        return len(inputStr)

#%% For file reading
# function to read all digits in directory and write them to numpy array
def write_digits(dir):
    files = listdir(dir)
    files.sort(key=sortByLength)
    n = len(files)
    m = 784
    num = np.zeros((n, m)) #for digit pixels
    
    # write matrix
    for i in range(n): 
        img = Image.open(dir + '\\' + files[i])
        temp = np.asarray(img.convert('L'))
        temp = temp.reshape(1, m)
        num[i,:]  = temp

    return(num)
    
#%% read data
trainnum = write_digits('digits_train')
testnum = write_digits('digits_test')
# answers
answ = pd.read_csv('digits_train_values.csv', header=0).as_matrix()
answ = np.asarray(answ)
answ = answ.reshape(answ.size, )


#%% Learning and testing

# random forest
from sklearn.ensemble import RandomForestClassifier as RFC

rf = RFC(n_estimators=150) # number of trees 
rf.fit(X=trainnum, y=answ) 

# cross validation
from sklearn import cross_validation
cross_validation.cross_val_score(rf, trainnum, answ, cv = 5) #5 sets

#%% Prediction

#exact prediction
test_classes = rf.predict(testnum)
answ_exact = pd.DataFrame({'id': range(8400), 'digit': list(test_classes)})[['id', 'digit']] # id before digit
answ_exact.head(3)
answ_exact.to_csv('Volkov_answ.csv', index=False)

#probability predictions
test_prob = rf.predict_proba(testnum)
answ_prob = pd.DataFrame(test_prob)
names = list(answ_prob.columns.values) #names for digits
answ_prob['id'] = range(8400) # add id
answ_prob = answ_prob[['id'] + names] # id has to be 1st
answ_exact.to_csv('Volkov_answ_prob.csv', index=False)