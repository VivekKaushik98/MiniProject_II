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


from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train,y_train.values.ravel())

y_pred=random_forest.predict(x_test)

random_forest.score(x_test,y_test)

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

cnf_matrix=confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cnf_matrix,classes=[0,1])

y_pred=random_forest.predict(x)

cnf_matrix=confusion_matrix(y,y_pred.round())

plot_confusion_matrix(cnf_matrix,classes=[0,1])
