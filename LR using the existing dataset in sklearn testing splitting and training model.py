import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#creating regressors
#importing prexisting dataset diabetes

diabetes=datasets.load_diabetes()

#slicing dataset and taking only one feature and only one label
diabetes_X=diabetes.data #index 2 e jei feature chilo seta 1 column e dibe arary of array hisabe
#train test splitting. full dataset er kichu data model train e use kora hoy baki data test korte
diabetes_X_train=diabetes_X[:-30] # ekhane diabetes_X er last 30 ta nicche
diabetes_X_test=diabetes_X[-30:] # ekhane diabetes_X er last 20 ta nicche

diabetes_Y_train=diabetes.target[:-30] #diabetes_X_train=diabetes_X[:-30]er corresponding feature. that means
#diabetes_X_train=diabetes_X[:-30] er jei feature tar label hocche eita tai[:-30]same
diabetes_Y_test=diabetes.target[-30:]

#x axis e features and y axis e labels thakbe
#creating linear model
model=linear_model.LinearRegression()
#linearmodel ke feedkorano hoise

#fitting data means data diye ekta line banabo jeta ei linear model e save hoye jabe
model.fit(diabetes_X_train,diabetes_Y_train) #training data fit koraye model train korano hoche
#checking model prediction
diabetes_Y_predicted=model.predict(diabetes_X_test) #testing our model with test data

print("Mean squared error is:", mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))#predicted values and actual values
print("Weight:",model.coef_)
print("intercept",model.intercept_)

#scatter plot
#plt.scatter(diabetes_X_test,diabetes_Y_test)
#plt.plot(diabetes_X_test,diabetes_Y_predicted)
#plt.show()

#Mean squared error is: 3035.0601152912686 when diabetes_X=diabetes.data[:, np.newaxis,2]
#Weight: [941.43097333]
#intercept 153.39713623331698


