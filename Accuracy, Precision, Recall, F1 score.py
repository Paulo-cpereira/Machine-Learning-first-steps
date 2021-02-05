import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X,y)
y_prev = model.predict(X)

print("Y previsto:",y_prev)
print("Recall:",recall_score(y, y_prev))
print("Precisão:",precision_score(y, y_prev))
print("Acc:",accuracy_score(y, y_prev))
print("F1:",f1_score(y, y_prev))
