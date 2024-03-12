# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Prasannalakshmi G
RegisterNumber: 212222240075

```
```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

x = data1. iloc[:, : -1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = "liblinear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test, y_pred)
print("Accuracy Score:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nClassificstion Report:\n",cr)

from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = [True, False])
cm_display.plot()
```

## Output:
## Accuracy Score and Classification Report:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118610231/b2a1f6e4-3c82-45ec-8bfc-879883bed4c4)

## Displaying:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118610231/c0f5a93f-3a34-41b2-9793-06d4382b158a)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
