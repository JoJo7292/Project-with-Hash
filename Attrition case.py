import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

emp_df1 = pd.read_excel(r'C:\Users\Chinonye Chibueze\Downloads\Hash Analytic\hash ass\Employees Attrition.xlsx', sheet_name='Existing employees')
emp_df2 = pd.read_excel(r'C:\Users\Chinonye Chibueze\Downloads\Hash Analytic\hash ass\Employees Attrition.xlsx', sheet_name='Employees who have left')

current = [1] * 11428
gone = [0] * 3571

emp_df1['current_employees'] = current
emp_df2['current_employees'] = gone

emp_df = pd.concat([emp_df1, emp_df2])

X = emp_df.iloc[:, :10].values
Y = emp_df.iloc[:, 10:11].values

from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
X[:, 9] = labelenc.fit_transform(X[:, 9])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('one_hot', OneHotEncoder(categories='auto'),[8])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:]

import seaborn as sns

plt.figure(figsize=(10,11))
sns.heatmap(emp_df.corr(), annot=True)
plt.plot()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
classifier.fit(X_train, Y_train)
Y_dtc = classifier.predict(X_test)

dt_acc = metrics.accuracy_score(Y_dtc, Y_test)
print('My Decision tree model accuracy is {} '.format(dt_acc*100))

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


kfold = KFold(n_splits=5, shuffle=False)
kfold.split(X)

accuracy_model = [] #initializing the accuracy of thr models to blank list.

for train_index, test_index in kfold.split(X): #to iterate over each train-test split
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    model = classifier.fit(X_train, Y_train)#training the model
    accuracy_model.append(accuracy_score(Y_test, model.predict(X_test), normalize=True)*100)
    
print('Our model accuracy is ', accuracy_model)


#predicting with Random forest
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=10)
random = RandomForestClassifier(n_estimators=100)
random.fit(X_train, Y_train)
RF_pred = random.predict(X_test)

rf_acc = accuracy_score(Y_test, RF_pred)
print('My Random forest model accuracy is {} '.format(rf_acc*100))


#KFold cross validation
from sklearn.model_selection import cross_val_score
rf_cv = RandomForestClassifier(n_estimators = 100)
scores = cross_val_score(rf_cv, X_train, Y_train, cv=10, scoring = 'accuracy')

print('Our Score is: ', scores)
print('mean is: ', scores.mean())
print('Standard deviation is: ', scores.std())


#Feature Importance
#importances = pd.DataFrame({'feature':X_train, 'importance':np.round(random.feature_importances_, 3)})
#importances= importances.sort_values('importance', ascending=False).set_index('feature')

importance = pd.Series(random.feature_importances_)
importnace = importance.nlargest(n=18)
importance.head()
importance.plot.bar()

































