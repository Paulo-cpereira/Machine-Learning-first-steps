import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80,random_state=25)      #train_size defines train datapoints%
                                                                                                #random_state forces the same datapoints selection (works like a seed)
model = LogisticRegression()

model.fit(X_train, y_train)
y_prev = model.predict(X_test)

print(model.score(X_test, y_test))
print("Recall:",recall_score(y_test, y_prev))
print("Precisão:",precision_score(y_test, y_prev))
print("Acc:",accuracy_score(y_test, y_prev))
print("F1:",f1_score(y_test, y_prev))

#print("whole dataset:", X.shape, y.shape)
#print("training set:", X_train.shape, y_train.shape)
#print("test set:", X_test.shape, y_test.shape)

