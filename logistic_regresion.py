#train a logistic regression model to predict wheter a flower is iris verginica or not
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris=datasets.load_iris() #loading iris datasets
#print(list(iris.keys()))
#taking only 3rd collumn from iris data(petal width)
X=iris["data"][:,3:] #slicing (features)
Y=(iris["target"]==2).astype(np.int)#labels. 2 karon sudhu iris verginica er jonno predict korchi
#astype verginica hole 1 hobe(true) ar na hole 0 dekhabe(false)
#print(X)

#train a logistic regression classifier
clf=LogisticRegression() #bulding a model
#fitting data into our model
clf.fit(X,Y)
example=clf.predict([[2.6]]) #petal width 1.6 er jonno predict korbe
print(example)

#using matplotlib to plot the visualization
#petal width er 1000 values nilam eta xaxis e plot korahobe
X_new=np.linspace(0,3,1000).reshape(-1,1) #linespace 0,3 er moddher 1000 points dibe ar reshape etake 1D array te nibe
#probabilty
#predict agerta just eta verginica hay ki na seta predict korto
#But pedict_proba actual value of probaility predict korbe etake y axis e plot korbo
X_prob=clf.predict_proba(X_new)
plt.plot(X_new,X_prob[:,1],"g-", label="virginica") #just ekta row dibe
plt.show()





