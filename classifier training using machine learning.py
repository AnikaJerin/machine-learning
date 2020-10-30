from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading dataset
iris=datasets.load_iris()
print(iris.DESCR) #description print korbe dataset er ( class:- Iris-Setosa,- Iris-Versicolour   - Iris-Virginica
# iris.data te data thake and iris .target e target thake
#prining features and labels
features = iris.data
labels = iris.target
print(features[0], labels[0])  # out put([5.1 3.5 1.4 0.2] 0). 0 mane eta Iris-Setosa

#training classifier
clf=KNeighborsClassifier() #classifier has been craeted
#fitting the classier
clf.fit(features,labels)
pred=clf.predict([[1,1,1,1]]) #sepallengthy,swidth,petal length,pwidth is given in a @D array. etar jonno predict korbe
print(pred)





