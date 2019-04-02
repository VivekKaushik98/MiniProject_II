import pandas as pd
import numpy as np
import keras

np.random.seed(2)

data=pd.read_csv('creditcard.csv')

data.head()

from sklearn.preprocessing import StandardScaler
data['normalizedAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Amount'],axis=1)

data.head()

data= data.drop(['Time'],axis=1)
data.head()

x= data.iloc[:,data.columns!='Class']
y= data.iloc[:,data.columns=='Class']

x.head()

y.head()

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test= train_test_split(x,y, test_size=0.3,random_state=0)


x_train.shape

x_test.shape

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

from keras.models import  Sequential
from keras.layers import Dense
from keras.layers import Dropout

model= Sequential([
    Dense(units=16,input_dim=29,activation='relu'),    
    Dense(units=26,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])

model.summary()


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=15,epochs=5)

score=model.evaluate(x_test,y_test)

print(score)

import matplotlib.pyplot as plt
import itertools
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


y_pred=model.predict(x_test)
y_test=pd.DataFrame(y_test)

cnf_matrix=confusion_matrix(y_test,y_pred.round())

print(cnf_matrix)

plot_confusion_matrix(cnf_matrix,classes=[0,1])

y_pred=model.predict(x)
y_expected=pd.DataFrame(y)
cnf_matrix1=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix1,classes=[0,1])
plt.show()

fraud_indices=np.array(data[data.Class==1].index)
number_records_fraud=len(fraud_indices)
print(number_records_fraud) 

normal_indices=data[data.Class==0].index

random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)
random_normal_indices=np.array(random_normal_indices)
print(len(random_normal_indices))

under_sample_indices=np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))

under_sample_data=data.iloc[under_sample_indices,:]

x_undersample=under_sample_data.iloc[:,under_sample_data.columns!='Class']
y_undersample=under_sample_data.iloc[:,under_sample_data.columns=='Class']

x_train,x_test,y_train,y_test=train_test_split(x_undersample,y_undersample,test_size=0.3)

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=15,epochs=5)

y_pred=model.predict(x_test)
y_expected=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1]

y_pred=model.predict(x)
y_expected=pd.DataFrame(y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])

%%bash
pip install -U imbalanced-learn

from imblearn.over_sampling import SMOTE

x_resample,y_resample=SMOTE().fit_sample(x,y.values.ravel())

y_resample=pd.DataFrame(y_resample)
x_resample=pd.DataFrame(x_resample)

x_train,x_test,y_train,y_test=train_test_split(x_resample,y_resample,test_size=0.3)

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=15,epochs=5)


y_pred=model.predict(x_test)
y_expected=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])

y_pred=model.predict(x)
y_expected=pd.DataFrame(y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])