# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# We have a very small dataset that we want to fit as accurately as possible using 
# all data points - so lets not split into Training/Test sets, use the entire set for Training
# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# the default 1st degree (y = mx + c) linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
prediction_at_lvl_6point5 = lin_reg.predict([[6.5]])

# plot the regression
y_pred_lin_reg = lin_reg.predict(X)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.show()

# 2nd polynomial regression (y = a + bx + cx^2)
# this really is still using LinearRegression model BUT the input feature matrix 
# will now contain the terms - constant (i.e 1) and X^2 in addition to X
from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree=2)
X_2_poly = poly_feature.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_2_poly, y)
prediction2_at_lvl_6point5 = lin_reg_2.predict(poly_feature.fit_transform([[6.5]]))

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_2_poly), color='blue')
plt.show()

# try 3rd degreee
poly_feature = PolynomialFeatures(degree=3)
X_3_poly = poly_feature.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_3_poly, y)
prediction2_at_lvl_6point5 = lin_reg_2.predict(poly_feature.fit_transform([[6.5]]))

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_3_poly), color='blue')
plt.show()