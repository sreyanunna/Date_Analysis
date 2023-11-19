#!/usr/bin/env python
# coding: utf-8

# # Analysis on Dates! 
# 
# Aim of this project :
# Making a fine-tune model, which is ready for deployment.
# 
# - Find out a Dataset, and compare at least two different algorithms and choose the best one
# -  Use suitable Data Preprocessing and Feature Selection/Engineering Methods
# -  Fine tune the model and hyper parameters and Finalise the Model
# -  Make the model deployment-ready by giving User-Input provision
# 
# Data set used : Date fruit dataset

# ![dates.jpg](attachment:dates.jpg)

# ### SECTIONS : 
# 
# 1. References 
# 2. Question 
# 3. About the dataset
# 4. Importing dataset
# 5. Impoting libraries
# 6. Metadata 
# 7. Splitting the dataset 
# 8. Implementing algorithms on dataset 
#     - Logistic regression 
#     - Decision tree classifier 
#     - Gradient Booster Classifier 
# 9. Standard Scalar 
# 10. Applying PCA 
# 11. EDA
# 12. User Input 
# 13. Conclusion 
# 
# 
# 
# 

# ### References
# 1. https://www.analyticsvidhya.com/blog/2021/04/how-the-gradient-boosting-algorithm-works/#:~:text=i)%20Gradient%20Boosting%20Algorithm%20is,cost%20function%20is%20Log%2DLoss.
# 2. https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/#:~:text=Gradient%20boosting%20classifiers%20are%20a,used%20when%20doing%20gradient%20boosting.
# 3. https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
# 4. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# 5. https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets/metadata
# 6. https://www.kaggle.com/code/casper6290/dateclassification
# 7. https://www.kaggle.com/code/darshansedani2205/date-fruit-dataset
# 8. https://www.askpython.com/python/examples/standardize-data-in-python
# 9. https://harish-reddy.medium.com/regularization-in-python-699cfbad8622#:~:text=Regularization%20helps%20to%20choose%20preferred,for%20many%20machine%20learning%20algorithms.
# 10. https://www.kdnuggets.com/2019/06/select-rows-columns-pandas.html
# 11. https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/
# 12. https://thenewstack.io/3-new-techniques-for-data-dimensionality-reduction-in-machine-learning/
# 13. https://towardsdatascience.com/11-dimensionality-reduction-techniques-you-should-know-in-2021-dcb9500d388b
# 14. https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
# 15. https://ieee-dataport.org/open-access/date-fruit-dataset-automated-harvesting-and-visual-yield-estimation
# 

# ## ABOUT THE DATASET 
# 
# - Name of dataset : **Date Fruit Datasets**
# - Link : https://www.muratkoklu.com/datasets/
# - **Abstract** : A great number of fruits are grown around the world, each of which has various types. The factors that determine the type of fruit are the external appearance features such as color, length, diameter, and shape. The external appearance of the fruits is a major determinant of the fruit type. Determining the variety of fruits by looking at their external appearance may necessitate expertise, which is time-consuming and requires great effort. The aim of this study is to classify the types of date fruit, that are, Barhee, Deglet Nour, Sukkary, Rotab Mozafati, Ruthana, Safawi, and Sagai by using three different machine learning methods. In accordance with this purpose, 898 images of seven different date fruit types were obtained via the computer vision system (CVS). Through image processing techniques, a total of 34 features, including morphological features, shape, and color, were extracted from these images. First, models were developed by using the logistic regression (LR) and artificial neural network (ANN) methods, which are among the machine learning methods. Performance results achieved with these methods are 91.0% and 92.2%, respectively. Then, with the stacking model created by combining these models, the performance result was increased to 92.8%. It has been concluded that machine learning methods can be applied successfully for the classification of date fruit types.

# ## Importing  Libraries 

# In[64]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import cmath
import seaborn as sns
import os
import itertools
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ## Importing the dataset 

# In[3]:


date = pd.read_csv("Date_Fruit_Datasets.csv",delimiter = ",")


# In[4]:


date


# ## METADATA 
# 

# In[5]:


date.info()


# this dataset in total contains 898 rows and 34 columns. Target varibale : Class

# In[6]:


date.isnull().values.any()


# No null values

# In[7]:


date.describe()


# Summary of the 5 major statistical values 

# ## Splitting the dataset 
# 
# Objective of this is to convert it in testing and training datasets as well as separating the target variable with independant variables 

# In[9]:


x=date.drop(['Class'],axis=1)
y=date['Class']


# In[10]:


# train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


# ## IMPLEMENTING ALGORITHMS ON DATASET 

# ### 1. LOGISTIC REGRESSION 
# Logistic regression is a machine learning classificiation algorithm that estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables.
# 
# **Why did I choose this ?** : 
# Because the target variable is a categorical variable. 

# ![logreg.png](attachment:logreg.png)

# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[12]:


logisticRegressor = LogisticRegression()


# In[13]:



model =logisticRegressor 
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


# In[14]:


accuracy_score(y_test,y_pred)


# #### The accuracy of logistic regression model on the date dataset (original ) is 60%

# ### 2. DECISION TREE MODEL 
# A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).
# 
# **Why did I choose this ?** 
# 
# Because I wanted to develop a classification system that predict or classify future observations based on a set of decision rules.

# ![dtm.png](attachment:dtm.png)

# In[15]:


# decision tree using gini index 
from sklearn.tree import DecisionTreeClassifier
dtree_gini=DecisionTreeClassifier(criterion='gini',random_state=0)
dtree_gini.fit(x_train,y_train)


# In[16]:


# prediction
y_pred=dtree_gini.predict(x_test)


# In[17]:


# evaluation
print('****************************Accuracy Score************************************')
ac=accuracy_score(y_test,y_pred)
print(ac)
print("**********************************Evaluation**********************************************")


evaluation = pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'])
print(evaluation)
print("**********************************Classification Report**********************************************")
classification=classification_report(y_test,y_pred)
print(classification)


# In[18]:


# decision tree using entropy index
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree.fit(x_train,y_train)


# In[19]:


# prediction
y_pred=dtree.predict(x_test)


# In[20]:


# evaluation
print('****************************Accuracy Score************************************')
ac=accuracy_score(y_test,y_pred)
print(ac)
print("**********************************Evaluation**********************************************")


evaluation = pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'])
print(evaluation)
print("**********************************Classification Report**********************************************")
classification_ent=classification_report(y_test,y_pred)
print(classification_ent)


# #### Thus, the decision tree model using entropy is more accurate that gini (82.22%)

# In[21]:


# What is the Depth of the Tree? How many Leaves are present in the same? [you may use get_depth() and get_n_leaves() methods].
print("**********************************Depth of the Tree**********************************************")
print(dtree.get_depth())
print("**********************************Number of Leaves**********************************************")
print(dtree.get_n_leaves())


# ### 3. GRADIENT BOOSTING CLASSIFIER
# 
# Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model.
# 
# **When to use Gradient Boosting classifier :**
# 
# -  Gradient Boosting Algorithm is generally used when we want to decrease the Bias error. 
# -  Gradient Boosting Algorithm can be used in regression as well as classification problems.

# ![gradient.png](attachment:gradient.png)

# In[23]:


err = []
for i in range(1,20):
    clf = GradientBoostingClassifier(n_estimators=i*10, learning_rate=1.0,max_depth=1)
    clf.fit(x_train,y_train)
    errpred = clf.predict(x_test)
    err.append(np.mean(errpred != y_test))


# In[25]:


GBC = GradientBoostingClassifier(n_estimators=200, learning_rate=10**-5,max_depth=20)
GBC.fit(x_train,y_train)
GBCpred = clf.predict(x_test)
print('Gradient Boosting' + '\n')
print(classification_report(y_test,GBCpred))


# Overall accuracy score of the gradient boosting classifier is 79%

# ## APPLYING STANDARD SCALAR : 
# 
# **What is Standard Scalar**
# 
# The main idea is to normalize/standardize i.e. μ = 0 and σ = 1 your features/variables/columns of X, individually, before applying any machine learning model.
# 

# In[35]:


scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
log_model=LogisticRegression()
log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)
accuracy_score(y_test,y_pred)


# The accuracy of model after applying standard scalar on the dataset(Using logistic regression)

# In[72]:


a=scaler.fit(x_train)


# ## APPLYING PCA 
# 
# **What is PCA?**
# 
# Principal component analysis (PCA) is the process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest.
# 
#  (PCA) simplifies the complexity in high-dimensional data while retaining trends and patterns.

# ![pca.jpg](attachment:pca.jpg)

# In[41]:


from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca.fit(date.iloc[:,:-1])
pca_array=pca.transform(date.iloc[:,:-1])
pca_dataset=pd.DataFrame(pca_array,columns=['PC1','PC2','PC3','PC4'])
pca_dataset.head()


# In[43]:


## PCA 
#Logistic regression PCA
X_train, X_test, y_train, y_test = train_test_split(pca_dataset, y, test_size=0.10, random_state=42)
log_model2=LogisticRegression()
log_model2.fit(X_train,y_train)
y_pred=log_model2.predict(X_test)
accuracy_score(y_test,y_pred)


# The accuracy of PCA via logistic regression is 64.5%

# ### INFERENCE 
# 
# The best fit model is the dataset standardized modeal with 85% accuracy using Logistic Regression

# ## EDA 

# In[93]:


plt.figure(figsize=(20,20))
_=sns.heatmap(date.corr(),cmap='PiYG')
plt.show();


# - The highest negative correlation is between SkewRG anf ALLdaub4RG
# - The highest positive correlation is between ALLdauband MEAN

# In[28]:


grouped=date.groupby(by='Class').mean()
grouped


# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
for i in grouped.columns:
    plt.figure(figsize=(8,6))
    sns.barplot(data=grouped,x=grouped.index,y=grouped[i])
    plt.show()


# In[31]:


plt.figure(figsize=(16,8))
sns.scatterplot(x=date['AREA'],y=date['PERIMETER'],hue=date['Class'],palette='YlGnBu',legend='auto');


# The largest dates are of class  SAFAVI and smallest being DOKOL 

# In[34]:


sns.countplot(data=date, x='Class', palette='YlGnBu');


# The most common date species are that of DOKOL. Least common being BERHI 

# ### USER INPUT 

# In[69]:


list1 = []
list1.append(int(input("Enter the area of date:")))
list1.append(float(input("Enter the perimeter of date:")))
list1.append(float(input("Enter the major axis of date:")))
list1.append(float(input("Enter the minor axis of date:")))
list1.append(float(input("Enter the eccentricity of date:")))
list1.append(float(input("Enter the eqdiasq of date:")))
list1.append(float(input("Enter the solidity of date:")))
list1.append(float(input("Enter the convex area of date:")))
list1.append(float(input("Enter the extent of date:")))
list1.append(float(input("Enter the aspect ratio of date:")))
list1.append(float(input("Enter the roundness of date:")))
list1.append(float(input("Enter the compactness of date:")))
list1.append(float(input("Enter the first shapefactor of date:")))
list1.append(float(input("Enter the second shapefactor of date:")))
list1.append(float(input("Enter the third shapefactor of date:")))
list1.append(float(input("Enter the fourth shapefactor of date:")))
list1.append(float(input("Enter the mean rr of date:")))
list1.append(float(input("Enter the mean rg of date:")))
list1.append(float(input("Enter the mean rb of date:")))
list1.append(float(input("Enter the standard deviation rr of date:")))
list1.append(float(input("Enter the standard deviation rg of date:")))
list1.append(float(input("Enter the standard deviation rb of date:")))
list1.append(float(input("Enter the skewness rr of date:")))
list1.append(float(input("Enter the skewness rg of date:")))
list1.append(float(input("Enter the skewness rb of date:")))
list1.append(float(input("Enter the kurtosis rb of date:")))
list1.append(float(input("Enter the kurtosis rg of date:")))
list1.append(float(input("Enter the entropy rr of date:")))
list1.append(float(input("Enter the entropy rb of date:")))
list1.append(float(input("Enter the entropy rg of date:")))
list1.append(float(input("Enter the ALLdaub4RR of date:")))
list1.append(float(input("Enter the ALLdaub4RG of date:")))
list1.append(float(input("Enter the ALLdaub4RB of date:")))


# In[70]:


b = pd.DataFrame(list1)
b = a.transform(b)


# ## CONCLUSION

# In conclusion, data preprocessing is an essential component of producing a machine learning model that is more accurate and efficient. At the same time, every dataset requires its own type of algorithm and pre processing techniques.  For example : for categorical classification - use logistic regression. 
# 
# EDA is an essential section to visualize and understand the data effectively. 
# This lab was very informative and provided an exhaustive analysis of the dataset I used. 
# 
# 
