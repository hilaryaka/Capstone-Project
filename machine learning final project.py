#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


# In[2]:


path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'


# In[3]:


df = pd.read_csv(path)
df.head()


# In[4]:


df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])


# In[5]:


df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)


# In[6]:


df_sydney_processed.drop('Date',axis=1,inplace=True)


# In[7]:


df_sydney_processed = df_sydney_processed.astype(float)


# In[8]:


features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']


# ## Q1) Use the train_test_split function to split the features and Y dataframes with a test_size of 0.2 and the random_state set to

# In[9]:


msk = np.random.rand(len(df))  < 0.2
train = df_sydney_processed[msk]
test = df_sydney_processed[~msk]


# In[11]:


plt.scatter(train.Humidity3pm, train.RainTomorrow,  color='blue')
plt.xlabel("Humidity3pm")
plt.ylabel("RainTomorrow")
plt.show()


# ## Q2) Create and train a Linear Regression model called LinearReg using the training data (x_train, y_train)

# In[14]:


from sklearn import linear_model
LinearReg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Humidity3pm']])
train_y = np.asanyarray(train[['RainTomorrow']])
LinearReg.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', LinearReg.coef_)
print ('Intercept: ',LinearReg.intercept_)


# ## Now use the predict method on the testing data (x_test) and save it to the array predictions

# In[15]:


LinearReg = linear_model.LinearRegression()

LinearReg.fit(train_x, train_y)


# In[26]:


predictions = LinearReg.predict(test_x)


# ## Q5) Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model

# In[20]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Humidity3pm']])
test_y = np.asanyarray(test[['RainTomorrow']])
test_y_ = LinearReg.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# ## Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function

# In[23]:


print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))


# In[24]:


print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))


# In[25]:


print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# # KNN

# In[27]:


from sklearn.neighbors import KNeighborsClassifier


# In[30]:


X = df[['MinTemp', 'MaxTemp','Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'Humidity3pm','Pressure3pm', 'Temp3pm', 'WindDir3pm']] .values  #.astype(float)
X[0:5]


# In[31]:


y = df['RainTomorrow'].values
y[0:5]


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[ ]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[ ]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# # Decision Tree

# In[49]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# ## Q9) Create and train a Decision Tree model called Tree using the training data (x_train, y_train)

# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)


# In[59]:


X = df_sydney_processed.drop(columns=["RainTomorrow"])
Y = df_sydney_processed["Humidity3pm"]


# In[60]:


X.head()


# In[61]:


Y.head()


# In[62]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=10)


# ## Q10) Now use the predict method on the testing data (x_test) and save it to the array predictions

# In[63]:


regression_tree = DecisionTreeRegressor(criterion = "mse")


# In[64]:


regression_tree.fit(X_train, Y_train)


# In[65]:


regression_tree.score(X_test, Y_test)


# ## Q11) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function

# In[66]:


prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)


# In[67]:


regression_tree = DecisionTreeRegressor(criterion = "mae")

regression_tree.fit(X_train, Y_train)

print(regression_tree.score(X_test, Y_test))

prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)


# # Logistic Regression

# In[ ]:




