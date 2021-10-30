import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

gender = {'Male': 1,'Female': 2}
veri = pd.read_csv("1.csv")
veri.head()

veri.Gender = [gender[item] for item in veri.Gender]

verison = veri.drop(['Index'],axis=1)
x = verison.iloc[:,1:3].values 
y = verison.iloc[:,0].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33, random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(y_pred)
print(y_test)