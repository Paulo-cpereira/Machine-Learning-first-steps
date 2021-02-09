import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'Male'
X = df[['Pclass','Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = LogisticRegression() # select the model
model.fit(X_train, y_train) # train the model
y_pred_proba = model.predict_proba(X_test) # predict the test data
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
auc = roc_auc_score(y_test,model.predict(X_test))

plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (LogisticRegression(), auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.legend(loc="lower right")
plt.show()   # Display
