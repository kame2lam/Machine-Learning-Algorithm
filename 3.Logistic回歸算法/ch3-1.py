from sklearn.linear_model import LogisticRegression
#Scikit-Learnh 的鳶尾花類數據集,是一個分類問題的數據集
from sklearn.datasets import load_iris
#載入鳶花數據集
X, y=load_iris(return_X_y=True)
#訓練模型
clf=LogisticRegression().fit(X, y)
#使用模型進行分數預測
x_=[[5.1,3.5,1.4,0.2],[5.9,3.,5.1,1.8]]
print(clf.predict(x_))
#性能評估器
print('性能:',clf.score(X,y))