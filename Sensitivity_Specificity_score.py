import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score

sensitivity_score = recall_score

def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true,y_pred)
    return r[0]

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

df['male'] = df['Sex'] == 'male'
X = df[['Pclass','male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80,random_state=25)      #train_size defines train datapoints% 
                                                                                                #random_state to obtain same values every run
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1] > 0.75   #using diferent values we increase/decrease our threshold

print("Sensitivity score(Positivos acertados):",sensitivity_score(y_test, y_pred))
print("Specificity score(Negativos acertados):",specificity_score(y_test, y_pred))
print("Precision score:",precision_score(y_test, y_pred))
print("Recall score:",recall_score(y_test, y_pred))
