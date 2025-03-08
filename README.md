# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.
   
## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NIRANJAN S
RegisterNumber:  24900209
*/
```

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
*/
```

## Output:
TO READ HEADS AND TAILS FILES
![265464941-636f3d3c-0e38-45d8-aa4c-5db428c68021](https://github.com/user-attachments/assets/0c8ad38b-9b50-4de4-9dee-712d50d52c60)

COMPARE DATASET
![265465326-f5fc907c-f184-4dd7-b5e4-a487a240c2a8](https://github.com/user-attachments/assets/8d27f278-91ce-42c4-af14-2b529ec6593c)

PREDICTED VALUE
![265465575-27c0cae3-fc3a-40ae-8b6d-ab668d74dba9](https://github.com/user-attachments/assets/cc0827a1-5fd9-4286-ad02-24ec233fa702)

GRAPH FOR TRAINING SET
![265465702-e182c03c-f168-4a5c-b1f7-dc0b5db7bf0d](https://github.com/user-attachments/assets/5166302f-48a3-4f09-95c7-2dcd8241468a)

GRAPH FOR TESTING SET
![265465895-a0b2c962-a9a7-4d1f-8b1c-11fbfae780bd](https://github.com/user-attachments/assets/e6b43e78-7a06-48f6-a27a-ab826fa1f38b)

ERRORS
![265465998-1e1134e2-e5bf-42c4-9dc9-04da16b41b6f](https://github.com/user-attachments/assets/d59e0dd8-2296-4c8e-bb2a-1c9f9ef28fa9)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
