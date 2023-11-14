"""              Simple Linear Regression               """

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and chosing dependent and independent variable for the analysis
dataset = pd.read_csv('Classification_college_lab/house_price.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set for traing the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#CREATES AN OBJECT OF LINEAR REGRESSION FOR THE VARIABLE 
regressor.fit(X_train, y_train)#HERE WE FIT THE DATA WICH NEEDS TO BE TRAIND USING THE LINER_REGRESSION THE INDEPENDENT AND THE DEPENDENT VARIABLE

# Predicting the Test set results just the numbers
y_pred = regressor.predict(X_test)#HERE WE PREDICT THE VALUES OF THE TEST DATA FROM THE THE ABOVE TRAINED DREGREESOR VARIABLE

# Visualising the Training set results 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()
 

    
"""
            #bonus    NOT PRESENT IN THE ORIGINAL          """

#Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))
#Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)

"""Therefore, the equation of our simple linear regression model is:

Salary=9345.94×YearsExperience+26816.19 

Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.
"""


#MY UNDER STANDING
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (ALL IN ONE)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

