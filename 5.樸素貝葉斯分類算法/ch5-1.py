from sklearn.datasets import load_iris
#載入樸素貝葉斯分類算法
from sklearn.naive_bayes import MultinomialNB
X, y=load_iris(return_X_y=True)
#訓練模型
clf=MultinomialNB().fit(X,y)
#使用模型進行分數預測
print(clf.predict(X))
print('性能:',clf.score(X,y))