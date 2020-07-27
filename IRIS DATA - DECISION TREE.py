import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

iris_df = pd.read_csv(r'C:\Users\Chinonye Chibueze\Downloads\Hash Analytic\hash ass\Iris.csv')

iris_df.isnull().count()
iris_df = iris_df.drop(["Id"], axis=1)

X = iris_df.iloc[:,0:4].values
Y = iris_df.iloc[:,4].values

plt.figure(figsize=(10,11))
sns.heatmap(iris_df.corr(), annot=True)
plt.plot()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


dtclass = DecisionTreeClassifier()
dtclass.fit(X_train, Y_train)
Y_pred = dtclass.predict(X_test)


dt_acc = metrics.accuracy_score(Y_pred, Y_test)
print('My Decision tree model accuracy is {} '.format(dt_acc*100))











