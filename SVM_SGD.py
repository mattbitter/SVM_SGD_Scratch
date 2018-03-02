import pandas as pd
import numpy as np
from sklearn import preprocessing
from random import randint
from random import choice
import random
from sklearn.model_selection import train_test_split

#import data - select only continuous columns
df = pd.read_csv("C:/Users/Splinta/GoogleDrive/Masters/Courses/CS498/Assignments/HW2/IncomeData2.csv")
dfconX = df[['AGE','FNLWGT','EDUNUM','CG','CL','HPW']].copy()
dfconY = df[['LABEL']]

dfconY.loc[dfconY['LABEL'] == " <=50K", 'LABEL'] = -1
dfconY.loc[dfconY['LABEL'] == ' >50K', 'LABEL'] = 1

#trian test split- seperate train and test then split train again into val and train.
#val and test have the same count of rows (10% of total each)
X_train, X_test, Y_train, Y_test = train_test_split(dfconX, dfconY, test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1111)
#training data
X = np.array(X_train)
Y = np.array(Y_train)

#scale data for mean 0 and unit variance and only scale on the training data.
#then apply that fit to the validation and test set
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
#X = preprocessing.scale(X)
X_val = scaler.transform(X_val)


Y_valp = np.array(Y_val)
#####This function selects a random number and if it is in the epoch hold out set then it picks
#another random number until it gets one not in the epoch hold out set
def rc():
    ran = choice(range(0, len(X) - 1))
    if ran == epochho:
        return rc()
    else:
        return ran

#set constants
b = 0
epochs = 51
lmbdalist = [.0001,.001,.01,.1,1]
accuracyStacked = pd.DataFrame(columns=lmbdalist)
magStacked = pd.DataFrame(columns=lmbdalist)
valaccuracy = pd.DataFrame(columns=lmbdalist)

#loop for all 4 lambdas
#https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/
#https://stats.stackexchange.com/questions/4608/gradient-of-hinge-loss

for lmbda in lmbdalist:
    w = np.zeros(len(X[0]))
    errors = []
    accuracy = []
    mag = []
    #loop for all epochs
    for epoch in range(1, epochs):
        error = 0
        #set epoch hold out list of 50
        epochho = random.sample(range(len(X) - 1), 50)
        #loop through steps in SGD
        for step in range(1,301):
            #set eta (step length)
            eta = 1/(50+epoch*.01 )
            #pick random vector in train set
            ran=rc()
            #calculate hinge loss
            if (Y[ran] * (np.dot(X[ran], w))) < 1:
                #update weights if incorrect classification and step
                w = w - (eta * (-(X[ran] * Y[ran]) + (lmbda * w)))
                #b = b - (eta * -Y[ran]) # did not use b because features were scaled to mean of 0
                error = error + 1
            else:
                #update weights if hinge loss was a correct classification and step
                w = w - (eta * (lmbda * w))

            #every 30 steps calculate accuracy and magnitude of w
            #using epoch hold out set of 50
            if step%30==0:
                accu = 0
                for i in range(0, len(epochho)):
                    if np.dot(X[epochho[i]], w) * Y[epochho[i]] > 0:
                        accu = accu + 1
                accuracy.append(accu)
                mag.append(np.sqrt(np.dot(w, w)))
        errors.append(error)

    #consolidating the accuracy and magnitudes
    accuracypercent = [(x / len(epochho)) * 100 for x in accuracy]
    accuracyStacked[lmbda] = accuracypercent
    magStacked[lmbda] = mag

    ########validation accuracy####################################
    #using validation set for all lambdas
    accuV = 0
    for i in range(0, len(X_val)):
        if np.dot(X_val[i], w) * Y_valp[i] > 0:
            accuV = accuV + 1
    valaccuracy[lmbda] = [(accuV / len(X_val))*100]

#do the plotting
axa = accuracyStacked.plot(title="Accuracy every 30 steps with varying lambda")
axa.set_xlabel("Steps (data points are incremented by 30 steps across all epochs)")
axa.set_ylabel("Accuracy")
axm = magStacked.plot(title="Magnitude every 30 steps with varying lambda")
axm.set_xlabel("Steps (data points are incremented by 30 steps across all epochs)")
axm.set_ylabel("Magnitude")
valaccuracy.plot(kind='bar',title="Validation Accuracy with varying lambda")


#############test accuracy########################################
Xt = np.append(X,X_val,axis=0)
Yt = np.append(Y,Y_valp,axis=0)
len(Xt)

w = np.zeros(len(X[0]))
epochs = 51
lmbda = .001

for epoch in range(1, epochs):
    error = 0
    for step in range(1,301):
        eta = 1/(50+epoch*.01 )
        #ran=rc()
        ran = randint(0, len(Xt)-1)
        if (Yt[ran] * (np.dot(Xt[ran], w))) < 1:
            w = w - (eta * (-(Xt[ran] * Yt[ran]) + (lmbda * w)))
            #b = b - (eta * -Y[ran])
            error = error + 1
        else:
            w = w - (eta * (lmbda * w))

    errors.append(error)

accuT = 0
X_test = scaler.transform(X_test)
Y_testp = np.array(Y_test)
for i in range(0, len(X_test)):
    if np.dot(X_test[i], w) * Y_testp[i] >0:
        accuT = accuT + 1
print ("Test accuracy with a lambda of ",lmbda," is: ", accuT/len(X_test))