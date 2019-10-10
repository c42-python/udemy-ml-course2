# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# y1 = dataset.iloc[:, -1:] -- not correct as this would return a dataframe, not a vector

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Transforms categorical non-numrical data to numerals
labelEncoder_X = LabelEncoder() # Transforms categorical non-numrical data to numerals
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3]) # i.e fit to the category column and also return the transformation
# Create new categorical columns for each of the distinct encoded category values
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy variable trap
X = X[:, 1:] # remove one of the encoded dummy variables

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - no need here
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.regression.linear_model as sm # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1) # this is for the constant portion of the linear equation 
                                                                # i.e (y = a + bx1) which can be thought of (y = ax0 + bx1)
                                                                # where x0 is 1
                                                                # statsmodel's OLS class does *not* do this by default, hence
                                                                
# we set our significance level at 0.05
                                                                
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # optimal features matrix - cycle 1
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() # we see that feature #2 has the highest p value (and > 0.05), so eliminate it

X_opt = X[:, [0, 1, 3, 4, 5]] # optimal features matrix - cycle 2
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() # we see that feature #1 has the highest p value (and > 0.05), so eliminate it

X_opt = X[:, [0, 3, 4, 5]] # optimal features matrix - cycle 3
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() # we see that feature #3 has the highest p value (and > 0.05), so eliminate it

X_opt = X[:, [0, 3, 5]] # optimal features matrix - cycle 4
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() # we see that feature #3 has the highest p value 0.06 (and > 0.05), so eliminate it

X_opt = X[:, [0, 3]] # optimal features matrix - cycle 5
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() # Finished.. none of the features have p-value above 0.05
print(regressor_OLS.summary())