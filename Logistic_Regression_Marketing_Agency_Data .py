#Importing Important Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#Reading the data from the csv file.
ad_data = pd.read_csv('advertising.csv')

#Checking the head, info and description of the data.
ad_data.head()
ad_data.info()
ad_data.describe()

#Visualizing the data
#Creating a histigram of the age.
sns.distplot(ad_data['Age'],kde=False,bins=30)

#Creating a jointplot showing Area Income versus Age.
sns.jointplot(y='Area Income',x='Age',data=ad_data)

#Creating a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
sns.jointplot(y='Daily Time Spent on Site',x='Age',data=ad_data,kind='kde', color='red')


#Creating a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'.
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')


#creating a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(ad_data,hue='Clicked on Ad')


#Spliting the data into training set and testing set using train_test_split.
from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=46)

#Importing the logistic regression module and fitting the data
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(X_train,y_train)

#Predictions and Evaluations
predictions = lg.predict(X_test)


#Creating a classification report for the model.
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

#
