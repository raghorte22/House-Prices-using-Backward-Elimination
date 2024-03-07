#!/usr/bin/env python
# coding: utf-8

# ###### House Prices using Backward Elimination
# Just started with machine learning. I have used backward Elimination to check the usefulness of dependent variables.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')

#importing dataset using pandas
dataset=pd.read_csv(r"D:\Data Science with AI\multiple linear regression\MLR\House_data.csv")

#to see my dataset is comprised of
dataset.head()


# In[2]:


#checking if any value is missing
print(dataset.isnull().any())


# In[3]:


dataset.head()


# In[4]:


dataset.columns


# In[5]:


dataset.info()


# In[6]:


#checking for categorical data
print(dataset.dtypes)


# In[7]:


# Check the actual column names in your DataFrame
print(dataset.columns)


# In[8]:


#droping the id and date columns
dataset = dataset.drop(['id' ,'date'], axis=1)


# In[12]:


# understanding the distribution with seaborn

sns.plotting_context("notebook",font_scale=2.5)
g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[]);


# In[13]:


#seprating independent and dependent variable
x= dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[15]:


from sklearn.linear_model import LinearRegression
regressor =LinearRegression()   #linar model= relationship between the input features and the output variable is assumed to be linear.
                 # LinearRegression=for modeling the relationship between a dependent variable (output) and one or more independent variables 
regressor.fit(x_train,y_train)    


# In[16]:


#predicting the test set results
y_pred =regressor.predict(x_test)


# In[17]:


#backward elimination 
import statsmodels.api as sm


# In[20]:


def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((21613,19)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
x_Modeled = backwardElimination(x_opt, SL)


# In[ ]:




