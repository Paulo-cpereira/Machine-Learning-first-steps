from sklearn.linear_model import LogisticRegression

n=int(input())
X=[]
for i in range (n):
    X.append([float(x) for x in input().split()])

y = [int(x)for x in input().split()]
datapoint = [float(x) for x in input().split()]


model = LogisticRegression()
model.fit(X,y)

previsao = model.predict([datapoint])

print(int(previsao))