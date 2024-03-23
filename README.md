# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start thr program
2. Import the required libraries and read the dataset
3. Initialize the program
4. Plot the points
5. End the program
## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JAYAKRISHNAN L B L 
RegisterNumber:  212222230052

```
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_predict, cross_val_score

# Load the dataset
data = pd.read_csv("Placement_Data.csv")

# Drop irrelevant columns
data.drop(['sl_no', 'salary'], axis=1, inplace=True)

# Handle missing values if any
data.fillna(method='ffill', inplace=True)

# Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the dataset into features and target variable
X = data.drop('status', axis=1)
y = data['status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Precision-Recall Curve
y_scores = cross_val_predict(model, X_train, y_train, cv=3, method='decision_function')
precision, recall, _ = precision_recall_curve(y_train, y_scores)
average_precision = average_precision_score(y_train, y_scores)
plt.plot(recall, precision, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Feature Importance Plot
feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Feature Importance")
plt.show()

# Cross-Validation Scores
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
plt.plot(np.arange(1, 6), cv_scores, marker='o')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores')
plt.xticks(np.arange(1, 6))
plt.show()

```
## Output:
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120232371/b2c60c66-3bcf-4f57-b7f0-76dcf6cd57de)
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120232371/9a26032c-2120-45e1-974d-32cd85018746)
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120232371/8b4a3a80-23de-450d-be99-c969b96dd5e5)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
